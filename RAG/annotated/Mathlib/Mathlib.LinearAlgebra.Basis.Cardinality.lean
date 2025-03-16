/--
Over any nontrivial ring, the existence of a finite spanning set implies that any basis is finite.
-/
lemma basis_finite_of_finite_spans (w : Set M) (hw : w.Finite) (s : span R w = ⊤) {ι : Type w}
    (b : Basis ι R M) : Finite ι := by
  classical
  haveI := hw.to_subtype
  cases nonempty_fintype w
  -- We'll work by contradiction, assuming `ι` is infinite.
  rw [← not_infinite_iff_finite]
  intro i
  -- Let `S` be the union of the supports of `x ∈ w` expressed as linear combinations of `b`.
  -- This is a finite set since `w` is finite.
  let S : Finset ι := Finset.univ.sup fun x : w => (b.repr x).support
  let bS : Set M := b '' S
  have h : ∀ x ∈ w, x ∈ span R bS := by
    intro x m
    rw [← b.linearCombination_repr x, span_image_eq_map_linearCombination, Submodule.mem_map]
    use b.repr x
    simp only [and_true, eq_self_iff_true, Finsupp.mem_supported]
    rw [Finset.coe_subset, ← Finset.le_iff_subset]
    exact Finset.le_sup (f := fun x : w ↦ (b.repr ↑x).support) (Finset.mem_univ (⟨x, m⟩ : w))
  -- Thus this finite subset of the basis elements spans the entire module.
  have k : span R bS = ⊤ := eq_top_iff.2 (le_trans s.ge (span_le.2 h))
  -- Now there is some `x : ι` not in `S`, since `ι` is infinite.
  obtain ⟨x, nm⟩ := Infinite.exists_not_mem_finset S
  -- However it must be in the span of the finite subset,
  have k' : b x ∈ span R bS := by
    rw [k]
    exact mem_top
  -- giving the desire contradiction.
  simp only [self_mem_span_image, Finset.mem_coe, bS] at k'
  exact nm k'


/-- Over any ring `R`, if `b` is a basis for a module `M`,
and `s` is a maximal linearly independent set,
then the union of the supports of `x ∈ s` (when written out in the basis `b`) is all of `b`.
-/
theorem union_support_maximal_linearIndependent_eq_range_basis {ι : Type w} (b : Basis ι R M)
    {κ : Type w'} (v : κ → M) (i : LinearIndependent R v) (m : i.Maximal) :
    ⋃ k, ((b.repr (v k)).support : Set ι) = Set.univ := by
  -- If that's not the case,
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Nontrivial R
    inst✝ : Module R M
    ι : Type w
    b : Basis ι R M
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    ⊢ Eq (Set.iUnion fun k => ↑(b.repr (v k)).support) Set.univ
  -/
  by_contra h
  simp only [← Ne.eq_def, ne_univ_iff_exists_not_mem, mem_iUnion, not_exists_not,
    Finsupp.mem_support_iff, Finset.mem_coe] at h
  -- We have some basis element `b b'` which is not in the support of any of the `v i`.
  /-
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Nontrivial R
    inst✝ : Module R M
    ι : Type w
    b : Basis ι R M
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    h : Exists fun a => ∀ (x : κ), Eq ((b.repr (v x)) a) 0
    ⊢ False
  -/
  obtain ⟨b', w⟩ := h
  -- Using this, we'll construct a linearly independent family strictly larger than `v`,
  -- by also using this `b b'`.
  /-
    case intro
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Nontrivial R
    inst✝ : Module R M
    ι : Type w
    b : Basis ι R M
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    b' : ι
    w : ∀ (x : κ), Eq ((b.repr (v x)) b') 0
    ⊢ False
  -/
  let v' : Option κ → M := fun o => o.elim (b b') v
  have r : range v ⊆ range v' := by
    rintro - ⟨k, rfl⟩
    use some k
    simp only [v', Option.elim_some]
  have r' : b b' ∉ range v := by
    rintro ⟨k, p⟩
    simpa [w] using congr_arg (fun m => (b.repr m) b') p
  have r'' : range v ≠ range v' := by
    intro e
    have p : b b' ∈ range v' := by
      use none
      simp only [v', Option.elim_none]
    rw [← e] at p
    exact r' p
  -- The key step in the proof is checking that this strictly larger family is linearly independent.
  have i' : LinearIndependent R ((↑) : range v' → M) := by
    apply LinearIndependent.to_subtype_range
    rw [linearIndependent_iff]
    intro l z
    rw [Finsupp.linearCombination_option] at z
    simp only [v', Option.elim'] at z
    change _ + Finsupp.linearCombination R v l.some = 0 at z
    -- We have some linear combination of `b b'` and the `v i`, which we want to show is trivial.
    -- We'll first show the coefficient of `b b'` is zero,
    -- by expressing the `v i` in the basis `b`, and using that the `v i` have no `b b'` term.
    have l₀ : l none = 0 := by
      rw [← eq_neg_iff_add_eq_zero] at z
      replace z := neg_eq_iff_eq_neg.mpr z
      apply_fun fun x => b.repr x b' at z
      simp only [repr_self, map_smul, mul_one, Finsupp.single_eq_same, Pi.neg_apply,
        Finsupp.smul_single', map_neg, Finsupp.coe_neg] at z
      erw [DFunLike.congr_fun (apply_linearCombination R (b.repr : M →ₗ[R] ι →₀ R) v l.some) b']
        at z
      simpa [Finsupp.linearCombination_apply, w] using z
    -- Then all the other coefficients are zero, because `v` is linear independent.
    have l₁ : l.some = 0 := by
      rw [l₀, zero_smul, zero_add] at z
      exact linearIndependent_iff.mp i _ z
    -- Finally we put those facts together to show the linear combination is trivial.
    ext (_ | a)
    · simp only [l₀, Finsupp.coe_zero, Pi.zero_apply]
    · erw [DFunLike.congr_fun l₁ a]
      simp only [Finsupp.coe_zero, Pi.zero_apply]
  /-
    case intro
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Nontrivial R
    inst✝ : Module R M
    ι : Type w
    b : Basis ι R M
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    b' : ι
    w : ∀ (x : κ), Eq ((b.repr (v x)) b') 0
    v' : Option κ → M := fun o => o.elim (b b') v
    r : HasSubset.Subset (Set.range v) (Set.range v')
    r' : Not (Membership.mem (Set.range v) (b b'))
    r'' : Ne (Set.range v) (Set.range v')
    i' : LinearIndependent R Subtype.val
    ⊢ False
  -/
  rw [LinearIndependent.Maximal] at m
  /-
    case intro
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Nontrivial R
    inst✝ : Module R M
    ι : Type w
    b : Basis ι R M
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : ∀ (s : Set M), LinearIndependent R Subtype.val → LE.le (Set.range v) s → E …
    b' : ι
    w : ∀ (x : κ), Eq ((b.repr (v x)) b') 0
    v' : Option κ → M := fun o => o.elim (b b') v
    r : HasSubset.Subset (Set.range v) (Set.range v')
    r' : Not (Membership.mem (Set.range v) (b b'))
    r'' : Ne (Set.range v) (Set.range v')
    i' : LinearIndependent R Subtype.val
    ⊢ False
  -/
  specialize m (range v') i' r
  /-
    case intro
    R : Type u
    M : Type v
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Nontrivial R
    inst✝ : Module R M
    ι : Type w
    b : Basis ι R M
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    b' : ι
    w : ∀ (x : κ), Eq ((b.repr (v x)) b') 0
    v' : Option κ → M := fun o => o.elim (b b') v
    r : HasSubset.Subset (Set.range v) (Set.range v')
    r' : Not (Membership.mem (Set.range v) (b b'))
    r'' : Ne (Set.range v) (Set.range v')
    i' : LinearIndependent R Subtype.val
    m : Eq (Set.range v) (Set.range v')
    ⊢ False
  -/
  exact r'' m
  /-
    🎉 no goals
  -/


/-- Over any ring `R`, if `b` is an infinite basis for a module `M`,
and `s` is a maximal linearly independent set,
then the cardinality of `b` is bounded by the cardinality of `s`.
-/
theorem infinite_basis_le_maximal_linearIndependent' {ι : Type w} (b : Basis ι R M) [Infinite ι]
    {κ : Type w'} (v : κ → M) (i : LinearIndependent R v) (m : i.Maximal) :
    Cardinal.lift.{w'} #ι ≤ Cardinal.lift.{w} #κ := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Nontrivial R
    inst✝¹ : Module R M
    ι : Type w
    b : Basis ι R M
    inst✝ : Infinite ι
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    ⊢ LE.le (Cardinal.lift.{w', w} (Cardinal.mk ι)) (Cardinal.lift.{w, w'} (Cardin …
  -/
  let Φ := fun k : κ => (b.repr (v k)).support
  have w₁ : #ι ≤ #(Set.range Φ) := by
    apply Cardinal.le_range_of_union_finset_eq_top
    exact union_support_maximal_linearIndependent_eq_range_basis b v i m
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Nontrivial R
    inst✝¹ : Module R M
    ι : Type w
    b : Basis ι R M
    inst✝ : Infinite ι
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    Φ : κ → Finset ι := fun k => (b.repr (v k)).support
    w₁ : LE.le (Cardinal.mk ι) (Cardinal.mk ↑(Set.range Φ))
    ⊢ LE.le (Cardinal.lift.{w', w} (Cardinal.mk ι)) (Cardinal.lift.{w, w'} (Cardin …
  -/
  have w₂ : Cardinal.lift.{w'} #(Set.range Φ) ≤ Cardinal.lift.{w} #κ := Cardinal.mk_range_le_lift
  /-
    R : Type u
    M : Type v
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Nontrivial R
    inst✝¹ : Module R M
    ι : Type w
    b : Basis ι R M
    inst✝ : Infinite ι
    κ : Type w'
    v : κ → M
    i : LinearIndependent R v
    m : i.Maximal
    Φ : κ → Finset ι := fun k => (b.repr (v k)).support
    w₁ : LE.le (Cardinal.mk ι) (Cardinal.mk ↑(Set.range Φ))
    w₂ : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.range Φ))) (Cardinal.lift …
    ⊢ LE.le (Cardinal.lift.{w', w} (Cardinal.mk ι)) (Cardinal.lift.{w, w'} (Cardin …
  -/
  exact (Cardinal.lift_le.mpr w₁).trans w₂
  /-
    🎉 no goals
  -/

-- (See `infinite_basis_le_maximal_linearIndependent'` for the more general version
-- where the index types can live in different universes.)

/-- Over any ring `R`, if `b` is an infinite basis for a module `M`,
and `s` is a maximal linearly independent set,
then the cardinality of `b` is bounded by the cardinality of `s`.
-/
theorem infinite_basis_le_maximal_linearIndependent {ι : Type w} (b : Basis ι R M) [Infinite ι]
    {κ : Type w} (v : κ → M) (i : LinearIndependent R v) (m : i.Maximal) : #ι ≤ #κ :=
  Cardinal.lift_le.mp (infinite_basis_le_maximal_linearIndependent' b v i m)


