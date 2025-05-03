/-- Let `n : ℤ` and `s` a subset of a commutative group `G` that is invariant under preimage for
the map `x ↦ x^n`. Then `s` is invariant under the pointwise action of the subgroup of elements
`g : G` such that `g^(n^j) = 1` for some `j : ℕ`. (This subgroup is called the Prüfer subgroup when
 `G` is the `Circle` and `n` is prime.) -/
@[to_additive
      "Let `n : ℤ` and `s` a subset of an additive commutative group `G` that is invariant
      under preimage for the map `x ↦ n • x`. Then `s` is invariant under the pointwise action of
      the additive subgroup of elements `g : G` such that `(n^j) • g = 0` for some `j : ℕ`.
      (This additive subgroup is called the Prüfer subgroup when `G` is the `AddCircle` and `n` is
      prime.)"]
theorem smul_eq_self_of_preimage_zpow_eq_self {G : Type*} [CommGroup G] {n : ℤ} {s : Set G}
    (hs : (fun x => x ^ n) ⁻¹' s = s) {g : G} {j : ℕ} (hg : g ^ n ^ j = 1) : g • s = s := by
  suffices ∀ {g' : G} (_ : g' ^ n ^ j = 1), g' • s ⊆ s by
    refine le_antisymm (this hg) ?_
    conv_lhs => rw [← smul_inv_smul g s]
    replace hg : g⁻¹ ^ n ^ j = 1 := by rw [inv_zpow, hg, inv_one]
    simpa only [le_eq_subset, set_smul_subset_set_smul_iff] using this hg
  /-
    G : Type u_1
    inst✝ : CommGroup G
    n : Int
    s : Set G
    hs : Eq (Set.preimage (fun x => HPow.hPow x n) s) s
    g : G
    j : Nat
    hg : Eq (HPow.hPow g (HPow.hPow n j)) 1
    ⊢ ∀ {g' : G}, Eq (HPow.hPow g' (HPow.hPow n j)) 1 → HasSubset.Subset (HSMul.hS …
  -/
  rw [(IsFixedPt.preimage_iterate hs j : (zpowGroupHom n)^[j] ⁻¹' s = s).symm]
  /-
    G : Type u_1
    inst✝ : CommGroup G
    n : Int
    s : Set G
    hs : Eq (Set.preimage (fun x => HPow.hPow x n) s) s
    g : G
    j : Nat
    hg : Eq (HPow.hPow g (HPow.hPow n j)) 1
    ⊢ ∀ {g' : G}, Eq (HPow.hPow g' (HPow.hPow n j)) 1 → HasSubset.Subset (HSMul.hS …
  -/
  rintro g' hg' - ⟨y, hy, rfl⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝ : CommGroup G
    n : Int
    s : Set G
    hs : Eq (Set.preimage (fun x => HPow.hPow x n) s) s
    g : G
    j : Nat
    hg : Eq (HPow.hPow g (HPow.hPow n j)) 1
    g' : G
    hg' : Eq (HPow.hPow g' (HPow.hPow n j)) 1
    y : G
    hy : Membership.mem (Set.preimage (Nat.iterate (fun x => HPow.hPow x n) j) s) y
    ⊢ Membership.mem (Set.preimage (Nat.iterate (fun x => HPow.hPow x n) j) s) ((f …
  -/
  change (zpowGroupHom n)^[j] (g' * y) ∈ s
  /-
    case intro.intro
    G : Type u_1
    inst✝ : CommGroup G
    n : Int
    s : Set G
    hs : Eq (Set.preimage (fun x => HPow.hPow x n) s) s
    g : G
    j : Nat
    hg : Eq (HPow.hPow g (HPow.hPow n j)) 1
    g' : G
    hg' : Eq (HPow.hPow g' (HPow.hPow n j)) 1
    y : G
    hy : Membership.mem (Set.preimage (Nat.iterate (fun x => HPow.hPow x n) j) s) y
    ⊢ Membership.mem s (Nat.iterate (⇑(zpowGroupHom n)) j (HMul.hMul g' y))
  -/
  replace hg' : (zpowGroupHom n)^[j] g' = 1 := by simpa [zpowGroupHom]
  /-
    case intro.intro
    G : Type u_1
    inst✝ : CommGroup G
    n : Int
    s : Set G
    hs : Eq (Set.preimage (fun x => HPow.hPow x n) s) s
    g : G
    j : Nat
    hg : Eq (HPow.hPow g (HPow.hPow n j)) 1
    g' y : G
    hy : Membership.mem (Set.preimage (Nat.iterate (fun x => HPow.hPow x n) j) s) y
    hg' : Eq (Nat.iterate (⇑(zpowGroupHom n)) j g') 1
    ⊢ Membership.mem s (Nat.iterate (⇑(zpowGroupHom n)) j (HMul.hMul g' y))
  -/
  rwa [iterate_map_mul, hg', one_mul]
  /-
    🎉 no goals
  -/

