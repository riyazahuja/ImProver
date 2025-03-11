instance finite [Finite Mˣ] [IsDomain R] : Finite (MulChar M R) := by
  have : Finite (Mˣ →* Rˣ) := by
    have : Fintype Mˣ := .ofFinite _
    let S := rootsOfUnity (Fintype.card Mˣ) R
    let F := Mˣ →* S
    have fF : Finite F := .of_injective _ DFunLike.coe_injective
    refine .of_surjective (fun f : F ↦ (Subgroup.subtype _).comp f) fun f ↦ ?_
    have H a : f a ∈ S := by simp only [mem_rootsOfUnity, ← map_pow, pow_card_eq_one, map_one, S]
    refine ⟨.codRestrict f S H, MonoidHom.ext fun _ ↦ ?_⟩
    simp only [MonoidHom.coe_comp, Subgroup.coeSubtype, Function.comp_apply,
      MonoidHom.codRestrict_apply]
  /-
    M : Type u_1
    R : Type u_2
    inst✝³ : CommMonoid M
    inst✝² : CommRing R
    inst✝¹ : Finite (Units M)
    inst✝ : IsDomain R
    this : Finite (MonoidHom (Units M) (Units R))
    ⊢ Finite (MulChar M R)
  -/
  exact .of_equiv _ MulChar.equivToUnitHom.symm
  /-
    🎉 no goals
  -/


lemma exists_apply_ne_one_iff_exists_monoidHom (a : Mˣ) :
    (∃ χ : MulChar M R, χ a ≠ 1) ↔ ∃ φ : Mˣ →* Rˣ, φ a ≠ 1 := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝¹ : CommMonoid M
    inst✝ : CommRing R
    a : Units M
    ⊢ Iff (Exists fun χ => Ne (χ ↑a) 1) (Exists fun φ => Ne (φ a) 1)
  -/
  refine ⟨fun ⟨χ, hχ⟩ ↦ ⟨χ.toUnitHom, ?_⟩, fun ⟨φ, hφ⟩ ↦ ⟨ofUnitHom φ, ?_⟩⟩
    /-
      case refine_1
      M : Type u_1
      R : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommRing R
      a : Units M
      x✝ : Exists fun χ => Ne (χ ↑a) 1
      χ : MulChar M R
      hχ : Ne (χ ↑a) 1
      ⊢ Ne (χ.toUnitHom a) 1
    -/
  · contrapose! hχ
    /-
      case refine_1
      M : Type u_1
      R : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommRing R
      a : Units M
      x✝ : Exists fun χ => Ne (χ ↑a) 1
      χ : MulChar M R
      hχ : Eq (χ.toUnitHom a) 1
      ⊢ Eq (χ ↑a) 1
    -/
    rwa [Units.ext_iff, coe_toUnitHom] at hχ
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      R : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommRing R
      a : Units M
      x✝ : Exists fun φ => Ne (φ a) 1
      φ : MonoidHom (Units M) (Units R)
      hφ : Ne (φ a) 1
      ⊢ Ne ((MulChar.ofUnitHom φ) ↑a) 1
    -/
  · contrapose! hφ
    /-
      case refine_2
      M : Type u_1
      R : Type u_2
      inst✝¹ : CommMonoid M
      inst✝ : CommRing R
      a : Units M
      x✝ : Exists fun φ => Ne (φ a) 1
      φ : MonoidHom (Units M) (Units R)
      hφ : Eq ((MulChar.ofUnitHom φ) ↑a) 1
      ⊢ Eq (φ a) 1
    -/
    simpa only [ofUnitHom_eq, equivToUnitHom_symm_coe, Units.val_eq_one] using hφ
    /-
      🎉 no goals
    -/


/-- If `M` is a finite commutative monoid and `R` is a ring that has enough roots of unity,
then for each `a ≠ 1` in `M`, there exists a multiplicative character `χ : M → R` such that
`χ a ≠ 1`. -/
theorem exists_apply_ne_one_of_hasEnoughRootsOfUnity [Nontrivial R] {a : M} (ha : a ≠ 1) :
    ∃ χ : MulChar M R, χ a ≠ 1 := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝⁴ : CommMonoid M
    inst✝³ : CommRing R
    inst✝² : Finite M
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units M))
    inst✝ : Nontrivial R
    a : M
    ha : Ne a 1
    ⊢ Exists fun χ => Ne (χ a) 1
  -/
  by_cases hu : IsUnit a
    /-
      case pos
      M : Type u_1
      R : Type u_2
      inst✝⁴ : CommMonoid M
      inst✝³ : CommRing R
      inst✝² : Finite M
      inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units M))
      inst✝ : Nontrivial R
      a : M
      ha : Ne a 1
      hu : IsUnit a
      ⊢ Exists fun χ => Ne (χ a) 1
    -/
  · refine (exists_apply_ne_one_iff_exists_monoidHom hu.unit).mpr ?_
    /-
      case pos
      M : Type u_1
      R : Type u_2
      inst✝⁴ : CommMonoid M
      inst✝³ : CommRing R
      inst✝² : Finite M
      inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units M))
      inst✝ : Nontrivial R
      a : M
      ha : Ne a 1
      hu : IsUnit a
      ⊢ Exists fun φ => Ne (φ hu.unit) 1
    -/
    refine CommGroup.exists_apply_ne_one_of_hasEnoughRootsOfUnity Mˣ R ?_
    /-
      case pos
      M : Type u_1
      R : Type u_2
      inst✝⁴ : CommMonoid M
      inst✝³ : CommRing R
      inst✝² : Finite M
      inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units M))
      inst✝ : Nontrivial R
      a : M
      ha : Ne a 1
      hu : IsUnit a
      ⊢ Ne hu.unit 1
    -/
    contrapose! ha
    /-
      case pos
      M : Type u_1
      R : Type u_2
      inst✝⁴ : CommMonoid M
      inst✝³ : CommRing R
      inst✝² : Finite M
      inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units M))
      inst✝ : Nontrivial R
      a : M
      hu : IsUnit a
      ha : Eq hu.unit 1
      ⊢ Eq a 1
    -/
    rw [← hu.unit_spec, ha, Units.val_eq_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_1
      R : Type u_2
      inst✝⁴ : CommMonoid M
      inst✝³ : CommRing R
      inst✝² : Finite M
      inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units M))
      inst✝ : Nontrivial R
      a : M
      ha : Ne a 1
      hu : Not (IsUnit a)
      ⊢ Exists fun χ => Ne (χ a) 1
    -/
  · exact ⟨1, by simpa only [map_nonunit _ hu] using zero_ne_one⟩
    /-
      🎉 no goals
    -/


/-- The group of `R`-valued multiplicative characters on a finite commutative monoid `M` is
(noncanonically) isomorphic to its unit group `Mˣ` when `R` is a ring that has enough roots
of unity. -/
lemma mulEquiv_units : Nonempty (MulChar M R ≃* Mˣ) :=
  ⟨mulEquivToUnitHom.trans
    (CommGroup.monoidHom_mulEquiv_of_hasEnoughRootsOfUnity Mˣ R).some⟩


/-- The cardinality of the group of `R`-valued multiplicative characters on a finite commutative
monoid `M` is the same as that of its unit group `Mˣ` when `R` is a ring that has enough roots
of unity. -/
lemma card_eq_card_units_of_hasEnoughRootsOfUnity : Nat.card (MulChar M R) = Nat.card Mˣ :=
  Nat.card_congr (mulEquiv_units M R).some.toEquiv


