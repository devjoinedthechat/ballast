# Walkthrough

Two tenants on one store. `acme` has three adapters over the same base model,
two of them successive runs of the same adapter. `partner` builds on one of
acme's through a grant. A signing key is set, so deletion proofs are signed.

```
export BALLAST_SIGNING_KEY=…
```

```
$ ballast --tenant acme commit support-v1 -m "support adapter, run 41" --ref support --metadata '{"run": 41}'
a7b571bab537  32 tensors  ref support

$ ballast --tenant acme commit support-v2 -m "run 42: retrained layers 3 and 9" --ref support
8e56ddf7bc74  32 tensors  ref support

$ ballast --tenant acme commit finance-v1 -m "finance adapter" --ref finance
bf31fbd196bb  32 tensors  ref finance

$ ballast --tenant acme stats
3 commits, 3 manifests, 66 blocks
12,582,912 logical bytes, 8,650,752 unique (1.45x dedup), 6,720,043 on disk (1.29x compression)

$ ballast --tenant acme log support
8e56ddf7bc74  leaf       run 42: retrained layers 3 and 9
a7b571bab537  leaf       support adapter, run 41

$ ballast --tenant acme diff a7b571bab537 8e56ddf7bc74
a7b571bab537 -> 8e56ddf7bc74
  tensors: 2 changed of 32 shared, 0 removed, 0 added
  relative change: 0.0499 overall, 0.2000 in the most changed tensor
  no fingerprints on both commits; behavioural effect unknown

$ ballast --tenant acme merge support@0.6 finance@0.4 --method ties --density 0.5 -m "blend" --ref blend
1008e6d71552  ties of 2 inputs  ref blend

$ ballast --tenant acme checkout blend -o ./blend-adapter
32 tensors -> ./blend-adapter

$ cat blend-adapter/ballast.json
{
  "base_model": "meta-llama/Llama-3.1-8B",
  "commit": "1008e6d715523daea683d1fbdcd3f8428a7ae90e7992e82e7adf8e38cc853c7a",
  "kind": "composite",
  "manifest": "b9ad3cb215b31dcb42aaa8f8fb1215a246156899f3465a936cad0ce56b75e208",
  "message": "support/finance blend",
  "metadata": {},
  "recipe": {
    "density": 0.5,
    "inputs": [
      {
        "manifest": "13a96fee350533627b01013bea8830cf20f142eaf50560f5a4ffe9112867b751",
        "tenant": "acme",
        "weight": 0.6
      },
      {
        "manifest": "8a93c0ba588346113c299112c619da48c7da439db5a3f10d25546a9ecbe2c361",
        "tenant": "acme",
        "weight": 0.4
      }
    ],
    "method": "ties"
  },
  "store": "…",
  "tenant": "acme"
}
$ ballast --tenant acme grant finance --to partner
partner may build views over acme:8a93c0ba5883

$ ballast --tenant partner commit finance-v1 -m "partner adapter"
f21a6cceb295  32 tensors  ref main

$ ballast --tenant partner merge acme:finance@0.5 main@0.5 -m "layered on acme" --ref layered
83b0537aae97  linear of 2 inputs  ref layered

$ ballast --tenant partner checkout layered -o ./layered
32 tensors -> ./layered

$ ballast --tenant acme revoke finance --to partner
revoked; 1 view(s) in 'partner' can no longer resolve

$ ballast --tenant partner checkout layered -o ./layered
cannot check out: composite 59ee760ddec6 cannot resolve: grant on acme:8a93c0ba5883 for 'partner' is missing or revoked
exit=2

$ ballast --tenant acme forget 8e56ddf7bc74 --reason "run 42 trained on withdrawn data"
tenant 'acme': 1 commits, 1 manifests, 2 blocks, 200,893 bytes
attestation f1c9c512e77c9b784afe2ff32d1c48c48f31bdc1cd15e817512bbe1c6e194d2d  (signed)
1 composite(s) now reference a deleted input and will refuse to resolve: acme:b9ad3cb215b31dc
verified

$ ballast --tenant acme checkout blend -o ./again
cannot check out: composite b9ad3cb215b3 cannot resolve: manifest 13a96fee3505 is missing for tenant 'acme'
exit=2

$ ballast --tenant acme verify f1c9c512e77c9b784afe2ff32d1c48c48f31bdc1cd15e817512bbe1c6e194d2d
tenant 'acme': 1 commits, 1 manifests, 2 blocks, 200,893 bytes
attestation f1c9c512e77c9b784afe2ff32d1c48c48f31bdc1cd15e817512bbe1c6e194d2d  (signed)
1 composite(s) now reference a deleted input and will refuse to resolve: acme:b9ad3cb215b31dc
verified

$ ballast --tenant acme fsck
acme: composite b9ad3cb215b3 references deleted input acme:13a96fee3505 (broken view)
exit=1
```

## What happened

**Run 42 changed two of thirty-two tensors, so it stored two blocks.** Every
other block was already there from run 41. The `stats` line says so: logical
bytes are what the manifests describe, unique bytes are what is actually held,
and the ratio between them is the deduplication.

**The blend stored nothing.** It is a view over `support` and `finance`, and it
resolved when it was checked out. The recipe — method, density, inputs and
weights — travels with the exported adapter in `ballast.json`, so anyone holding
the directory can see what it was made from.

**The partner's layered view resolved through a grant, and stopped resolving
the moment the grant was revoked.** Nothing was copied to the partner's tenant:
their view read acme's blocks through the grant on every checkout. Revoking it
left nothing behind to clean up.

**Forgetting run 42 removed exactly the two blocks nothing else referenced.**
The proof names the commit, the manifest and the blocks, records the attestation
under a signature, and names the composite it broke. `verify` re-checked it
afterwards from the attestation alone.

**The blend is still in history, and refuses to resolve.** That is the correct
state. A view over a deleted input has nothing to show; materialising it would
have meant producing a matrix with a deleted contribution smeared through it.
`fsck` reports it, and will keep reporting it until the blend is forgotten too.
