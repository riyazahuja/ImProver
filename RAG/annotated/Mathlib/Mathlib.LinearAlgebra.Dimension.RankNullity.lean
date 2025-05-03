/--
`HasRankNullity.{u}` is a class of rings satisfying
1. Every `R`-module `M : Type u` has a linear independent subset of cardinality `Module.rank R M`.
2. `rank (M ⧸ N) + rank N = rank M` for every `R`-module `M : Type u` and every `N : Submodule R M`.

Usually such a ring satisfies `HasRankNullity.{w}` for all universes `w`, and the universe
argument is there because of technical limitations to universe polymorphism.

See `DivisionRing.hasRankNullity` and `IsDomain.hasRankNullity`.
-/
@[pp_with_univ]
class HasRankNullity (R : Type v) [inst : Ring R] : Prop where
  exists_set_linearIndependent : ∀ (M : Type u) [AddCommGroup M] [Module R M],
    ∃ s : Set M, #s = Module.rank R M ∧ LinearIndependent (ι := s) R Subtype.val
  rank_quotient_add_rank : ∀ {M : Type u} [AddCommGroup M] [Module R M] (N : Submodule R M),
    Module.rank R (M ⧸ N) + Module.rank R N = Module.rank R M


lemma Submodule.rank_quotient_add_rank (N : Submodule R M) :
    Module.rank R (M ⧸ N) + Module.rank R N = Module.rank R M :=
  HasRankNullity.rank_quotient_add_rank N


variable (R M) in
lemma exists_set_linearIndependent :
    ∃ s : Set M, #s = Module.rank R M ∧ LinearIndependent (ι := s) R Subtype.val :=
  HasRankNullity.exists_set_linearIndependent M


variable (R) in
theorem nontrivial_of_hasRankNullity : Nontrivial R := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : HasRankNullity.{u, u_1} R
    ⊢ Nontrivial R
  -/
  refine (subsingleton_or_nontrivial R).resolve_left fun H ↦ ?_
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : HasRankNullity.{u, u_1} R
    H : Subsingleton R
    ⊢ False
  -/
  have := rank_quotient_add_rank (R := R) (M := PUnit) ⊥
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : HasRankNullity.{u, u_1} R
    H : Subsingleton R
    this : Eq (HAdd.hAdd (Module.rank R (HasQuotient.Quotient PUnit.{u + 1} Bot.bo …
    ⊢ False
  -/
  simp [one_add_one_eq_two] at this
  /-
    🎉 no goals
  -/


theorem LinearMap.lift_rank_range_add_rank_ker (f : M →ₗ[R] M') :
    lift.{u} (Module.rank R (LinearMap.range f)) + lift.{v} (Module.rank R (LinearMap.ker f)) =
      lift.{v} (Module.rank R M) := by
  /-
    R : Type u_1
    M : Type u
    M' : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    inst✝ : HasRankNullity.{u, u_1} R
    f : LinearMap (RingHom.id R) M M'
    ⊢ Eq (HAdd.hAdd (Cardinal.lift.{u, v} (Module.rank R (Subtype fun x => Members …
  -/
  haveI := fun p : Submodule R M => Classical.decEq (M ⧸ p)
  /-
    R : Type u_1
    M : Type u
    M' : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    inst✝ : HasRankNullity.{u, u_1} R
    f : LinearMap (RingHom.id R) M M'
    this : (p : Submodule R M) → DecidableEq (HasQuotient.Quotient M p)
    ⊢ Eq (HAdd.hAdd (Cardinal.lift.{u, v} (Module.rank R (Subtype fun x => Members …
  -/
  rw [← f.quotKerEquivRange.lift_rank_eq, ← lift_add, rank_quotient_add_rank]
  /-
    🎉 no goals
  -/


/-- The **rank-nullity theorem** -/
theorem LinearMap.rank_range_add_rank_ker (f : M →ₗ[R] M₁) :
    Module.rank R (LinearMap.range f) + Module.rank R (LinearMap.ker f) = Module.rank R M := by
  /-
    R : Type u_1
    M M₁ : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M
    inst✝¹ : Module R M₁
    inst✝ : HasRankNullity.{u, u_1} R
    f : LinearMap (RingHom.id R) M M₁
    ⊢ Eq (HAdd.hAdd (Module.rank R (Subtype fun x => Membership.mem (LinearMap.ran …
  -/
  haveI := fun p : Submodule R M => Classical.decEq (M ⧸ p)
  /-
    R : Type u_1
    M M₁ : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M
    inst✝¹ : Module R M₁
    inst✝ : HasRankNullity.{u, u_1} R
    f : LinearMap (RingHom.id R) M M₁
    this : (p : Submodule R M) → DecidableEq (HasQuotient.Quotient M p)
    ⊢ Eq (HAdd.hAdd (Module.rank R (Subtype fun x => Membership.mem (LinearMap.ran …
  -/
  rw [← f.quotKerEquivRange.rank_eq, rank_quotient_add_rank]
  /-
    🎉 no goals
  -/


theorem LinearMap.lift_rank_eq_of_surjective {f : M →ₗ[R] M'} (h : Surjective f) :
    lift.{v} (Module.rank R M) =
      lift.{u} (Module.rank R M') + lift.{v} (Module.rank R (LinearMap.ker f)) := by
  /-
    R : Type u_1
    M : Type u
    M' : Type v
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M
    inst✝¹ : Module R M'
    inst✝ : HasRankNullity.{u, u_1} R
    f : LinearMap (RingHom.id R) M M'
    h : Function.Surjective ⇑f
    ⊢ Eq (Cardinal.lift.{v, u} (Module.rank R M)) (HAdd.hAdd (Cardinal.lift.{u, v} …
  -/
  rw [← lift_rank_range_add_rank_ker f, ← rank_range_of_surjective f h]
  /-
    🎉 no goals
  -/


theorem LinearMap.rank_eq_of_surjective {f : M →ₗ[R] M₁} (h : Surjective f) :
    Module.rank R M = Module.rank R M₁ + Module.rank R (LinearMap.ker f) := by
  /-
    R : Type u_1
    M M₁ : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup M₁
    inst✝² : Module R M
    inst✝¹ : Module R M₁
    inst✝ : HasRankNullity.{u, u_1} R
    f : LinearMap (RingHom.id R) M M₁
    h : Function.Surjective ⇑f
    ⊢ Eq (Module.rank R M) (HAdd.hAdd (Module.rank R M₁) (Module.rank R (Subtype f …
  -/
  rw [← rank_range_add_rank_ker f, ← rank_range_of_surjective f h]
  /-
    🎉 no goals
  -/


theorem exists_linearIndependent_of_lt_rank [StrongRankCondition R]
    {s : Set M} (hs : LinearIndependent (ι := s) R Subtype.val) :
    ∃ t, s ⊆ t ∧ #t = Module.rank R M ∧ LinearIndependent (ι := t) R Subtype.val := by
  /-
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    s : Set M
    hs : LinearIndependent R Subtype.val
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (Cardinal.mk ↑t) (Module …
  -/
  obtain ⟨t, ht, ht'⟩ := exists_set_linearIndependent R (M ⧸ Submodule.span R s)
  /-
    case intro.intro
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    s : Set M
    hs : LinearIndependent R Subtype.val
    t : Set (HasQuotient.Quotient M (Submodule.span R s))
    ht : Eq (Cardinal.mk ↑t) (Module.rank R (HasQuotient.Quotient M (Submodule.spa …
    ht' : LinearIndependent R Subtype.val
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (Cardinal.mk ↑t) (Module …
  -/
  choose sec hsec using Submodule.Quotient.mk_surjective (Submodule.span R s)
  /-
    case intro.intro
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    s : Set M
    hs : LinearIndependent R Subtype.val
    t : Set (HasQuotient.Quotient M (Submodule.span R s))
    ht : Eq (Cardinal.mk ↑t) (Module.rank R (HasQuotient.Quotient M (Submodule.spa …
    ht' : LinearIndependent R Subtype.val
    sec : HasQuotient.Quotient M (Submodule.span R s) → M
    hsec : ∀ (b : HasQuotient.Quotient M (Submodule.span R s)), Eq (Submodule.Quot …
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (Cardinal.mk ↑t) (Module …
  -/
  have hsec' : Submodule.Quotient.mk ∘ sec = _root_.id := funext hsec
  have hst : Disjoint s (sec '' t) := by
    rw [Set.disjoint_iff]
    rintro _ ⟨hxs, ⟨x, hxt, rfl⟩⟩
    apply ht'.ne_zero ⟨x, hxt⟩
    rw [Subtype.coe_mk, ← hsec x, Submodule.Quotient.mk_eq_zero]
    exact Submodule.subset_span hxs
  /-
    case intro.intro
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    s : Set M
    hs : LinearIndependent R Subtype.val
    t : Set (HasQuotient.Quotient M (Submodule.span R s))
    ht : Eq (Cardinal.mk ↑t) (Module.rank R (HasQuotient.Quotient M (Submodule.spa …
    ht' : LinearIndependent R Subtype.val
    sec : HasQuotient.Quotient M (Submodule.span R s) → M
    hsec : ∀ (b : HasQuotient.Quotient M (Submodule.span R s)), Eq (Submodule.Quot …
    hsec' : Eq (Function.comp Submodule.Quotient.mk sec) id
    hst : Disjoint s (Set.image sec t)
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (Eq (Cardinal.mk ↑t) (Module …
  -/
  refine ⟨s ∪ sec '' t, subset_union_left, ?_, ?_⟩
  · rw [Cardinal.mk_union_of_disjoint hst, Cardinal.mk_image_eq, ht,
      ← rank_quotient_add_rank (Submodule.span R s), add_comm, rank_span_set hs]
    /-
      case intro.intro.refine_1
      R : Type u_1
      M : Type u
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : HasRankNullity.{u, u_1} R
      inst✝ : StrongRankCondition R
      s : Set M
      hs : LinearIndependent R Subtype.val
      t : Set (HasQuotient.Quotient M (Submodule.span R s))
      ht : Eq (Cardinal.mk ↑t) (Module.rank R (HasQuotient.Quotient M (Submodule.spa …
      ht' : LinearIndependent R Subtype.val
      sec : HasQuotient.Quotient M (Submodule.span R s) → M
      hsec : ∀ (b : HasQuotient.Quotient M (Submodule.span R s)), Eq (Submodule.Quot …
      hsec' : Eq (Function.comp Submodule.Quotient.mk sec) id
      hst : Disjoint s (Set.image sec t)
      ⊢ Function.Injective sec
    -/
    exact HasLeftInverse.injective ⟨Submodule.Quotient.mk, hsec⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      M : Type u
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : HasRankNullity.{u, u_1} R
      inst✝ : StrongRankCondition R
      s : Set M
      hs : LinearIndependent R Subtype.val
      t : Set (HasQuotient.Quotient M (Submodule.span R s))
      ht : Eq (Cardinal.mk ↑t) (Module.rank R (HasQuotient.Quotient M (Submodule.spa …
      ht' : LinearIndependent R Subtype.val
      sec : HasQuotient.Quotient M (Submodule.span R s) → M
      hsec : ∀ (b : HasQuotient.Quotient M (Submodule.span R s)), Eq (Submodule.Quot …
      hsec' : Eq (Function.comp Submodule.Quotient.mk sec) id
      hst : Disjoint s (Set.image sec t)
      ⊢ LinearIndependent R Subtype.val
    -/
  · apply LinearIndependent.union_of_quotient Submodule.subset_span hs
    rwa [Function.comp_def, linearIndependent_image (hsec'.symm ▸ injective_id).injOn.image_of_comp,
      ← image_comp, hsec', image_id]


/-- Given a family of `n` linearly independent vectors in a space of dimension `> n`, one may extend
the family by another vector while retaining linear independence. -/
theorem exists_linearIndependent_cons_of_lt_rank [StrongRankCondition R] {n : ℕ} {v : Fin n → M}
    (hv : LinearIndependent R v) (h : n < Module.rank R M) :
    ∃ (x : M), LinearIndependent R (Fin.cons x v) := by
  /-
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    n : Nat
    v : Fin n → M
    hv : LinearIndependent R v
    h : LT.lt (↑n) (Module.rank R M)
    ⊢ Exists fun x => LinearIndependent R (Fin.cons x v)
  -/
  obtain ⟨t, h₁, h₂, h₃⟩ := exists_linearIndependent_of_lt_rank hv.to_subtype_range
  have : range v ≠ t := by
    refine fun e ↦ h.ne ?_
    rw [← e, ← lift_injective.eq_iff, mk_range_eq_of_injective hv.injective] at h₂
    simpa only [mk_fintype, Fintype.card_fin, lift_natCast, lift_id'] using h₂
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    n : Nat
    v : Fin n → M
    hv : LinearIndependent R v
    h : LT.lt (↑n) (Module.rank R M)
    t : Set M
    h₁ : HasSubset.Subset (Set.range v) t
    h₂ : Eq (Cardinal.mk ↑t) (Module.rank R M)
    h₃ : LinearIndependent R Subtype.val
    this : Ne (Set.range v) t
    ⊢ Exists fun x => LinearIndependent R (Fin.cons x v)
  -/
  obtain ⟨x, hx, hx'⟩ := nonempty_of_ssubset (h₁.ssubset_of_ne this)
  exact ⟨x, (linearIndependent_subtype_range (Fin.cons_injective_iff.mpr ⟨hx', hv.injective⟩)).mp
    (h₃.mono (Fin.range_cons x v ▸ insert_subset hx h₁))⟩


/-- Given a family of `n` linearly independent vectors in a space of dimension `> n`, one may extend
the family by another vector while retaining linear independence. -/
theorem exists_linearIndependent_snoc_of_lt_rank [StrongRankCondition R] {n : ℕ} {v : Fin n → M}
    (hv : LinearIndependent R v) (h : n < Module.rank R M) :
    ∃ (x : M), LinearIndependent R (Fin.snoc v x) := by
  /-
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    n : Nat
    v : Fin n → M
    hv : LinearIndependent R v
    h : LT.lt (↑n) (Module.rank R M)
    ⊢ Exists fun x => LinearIndependent R (Fin.snoc v x)
  -/
  simp only [Fin.snoc_eq_cons_rotate]
  /-
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    n : Nat
    v : Fin n → M
    hv : LinearIndependent R v
    h : LT.lt (↑n) (Module.rank R M)
    ⊢ Exists fun x => LinearIndependent R fun i => Fin.cons x v ((finRotate (HAdd. …
  -/
  have ⟨x, hx⟩ := exists_linearIndependent_cons_of_lt_rank hv h
  /-
    R : Type u_1
    M : Type u
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : HasRankNullity.{u, u_1} R
    inst✝ : StrongRankCondition R
    n : Nat
    v : Fin n → M
    hv : LinearIndependent R v
    h : LT.lt (↑n) (Module.rank R M)
    x : M
    hx : LinearIndependent R (Fin.cons x v)
    ⊢ Exists fun x => LinearIndependent R fun i => Fin.cons x v ((finRotate (HAdd. …
  -/
  exact ⟨x, hx.comp _ (finRotate _).injective⟩
  /-
    🎉 no goals
  -/


/-- Given a nonzero vector in a space of dimension `> 1`, one may find another vector linearly
independent of the first one. -/
theorem exists_linearIndependent_pair_of_one_lt_rank [StrongRankCondition R]
    [NoZeroSMulDivisors R M] (h : 1 < Module.rank R M) {x : M} (hx : x ≠ 0) :
    ∃ y, LinearIndependent R ![x, y] := by
  /-
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : NoZeroSMulDivisors R M
    h : LT.lt 1 (Module.rank R M)
    x : M
    hx : Ne x 0
    ⊢ Exists fun y => LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matr …
  -/
  obtain ⟨y, hy⟩ := exists_linearIndependent_snoc_of_lt_rank (linearIndependent_unique ![x] hx) h
  /-
    case intro
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : NoZeroSMulDivisors R M
    h : LT.lt 1 (Module.rank R M)
    x : M
    hx : Ne x 0
    y : M
    hy : LinearIndependent R (Fin.snoc (Matrix.vecCons x Matrix.vecEmpty) y)
    ⊢ Exists fun y => LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matr …
  -/
  have : Fin.snoc ![x] y = ![x, y] := by simp [Fin.snoc, ← List.ofFn_inj]
  /-
    case intro
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : NoZeroSMulDivisors R M
    h : LT.lt 1 (Module.rank R M)
    x : M
    hx : Ne x 0
    y : M
    hy : LinearIndependent R (Fin.snoc (Matrix.vecCons x Matrix.vecEmpty) y)
    this : Eq (Fin.snoc (Matrix.vecCons x Matrix.vecEmpty) y) (Matrix.vecCons x (M …
    ⊢ Exists fun y => LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matr …
  -/
  rw [this] at hy
  /-
    case intro
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : NoZeroSMulDivisors R M
    h : LT.lt 1 (Module.rank R M)
    x : M
    hx : Ne x 0
    y : M
    hy : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty))
    this : Eq (Fin.snoc (Matrix.vecCons x Matrix.vecEmpty) y) (Matrix.vecCons x (M …
    ⊢ Exists fun y => LinearIndependent R (Matrix.vecCons x (Matrix.vecCons y Matr …
  -/
  exact ⟨y, hy⟩
  /-
    🎉 no goals
  -/


theorem Submodule.exists_smul_not_mem_of_rank_lt {N : Submodule R M}
    (h : Module.rank R N < Module.rank R M) : ∃ m : M, ∀ r : R, r ≠ 0 → r • m ∉ N := by
  have : Module.rank R (M ⧸ N) ≠ 0 := by
    intro e
    rw [← rank_quotient_add_rank N, e, zero_add] at h
    exact h.ne rfl
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    N : Submodule R M
    h : LT.lt (Module.rank R (Subtype fun x => Membership.mem N x)) (Module.rank R …
    this : Ne (Module.rank R (HasQuotient.Quotient M N)) 0
    ⊢ Exists fun m => ∀ (r : R), Ne r 0 → Not (Membership.mem N (HSMul.hSMul r m))
  -/
  rw [ne_eq, rank_eq_zero_iff, (Submodule.Quotient.mk_surjective N).forall] at this
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    N : Submodule R M
    h : LT.lt (Module.rank R (Subtype fun x => Membership.mem N x)) (Module.rank R …
    this : Not (∀ (x : M), Exists fun a => And (Ne a 0) (Eq (HSMul.hSMul a (Submod …
    ⊢ Exists fun m => ∀ (r : R), Ne r 0 → Not (Membership.mem N (HSMul.hSMul r m))
  -/
  push_neg at this
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    N : Submodule R M
    h : LT.lt (Module.rank R (Subtype fun x => Membership.mem N x)) (Module.rank R …
    this : Exists fun x => ∀ (a : R), Ne a 0 → Ne (HSMul.hSMul a (Submodule.Quotie …
    ⊢ Exists fun m => ∀ (r : R), Ne r 0 → Not (Membership.mem N (HSMul.hSMul r m))
  -/
  simp_rw [← N.mkQ_apply, ← map_smul, N.mkQ_apply, ne_eq, Submodule.Quotient.mk_eq_zero] at this
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    N : Submodule R M
    h : LT.lt (Module.rank R (Subtype fun x => Membership.mem N x)) (Module.rank R …
    this : Exists fun x => ∀ (a : R), Not (Eq a 0) → Not (Membership.mem N (HSMul. …
    ⊢ Exists fun m => ∀ (r : R), Ne r 0 → Not (Membership.mem N (HSMul.hSMul r m))
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem Submodule.rank_sup_add_rank_inf_eq (s t : Submodule R M) :
    Module.rank R (s ⊔ t : Submodule R M) + Module.rank R (s ⊓ t : Submodule R M) =
    Module.rank R s + Module.rank R t := by
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    s t : Submodule R M
    ⊢ Eq (HAdd.hAdd (Module.rank R (Subtype fun x => Membership.mem (Max.max s t)  …
  -/
  conv_rhs => enter [2]; rw [show t = (s ⊔ t) ⊓ t by simp]
  rw [← rank_quotient_add_rank ((s ⊓ t).comap s.subtype),
    ← rank_quotient_add_rank (t.comap (s ⊔ t).subtype),
    (quotientInfEquivSupQuotient s t).rank_eq,
    (equivSubtypeMap s (comap _ (s ⊓ t))).rank_eq, Submodule.map_comap_subtype,
    (equivSubtypeMap (s ⊔ t) (comap _ t)).rank_eq, Submodule.map_comap_subtype,
    ← inf_assoc, inf_idem, add_right_comm]


theorem Submodule.rank_add_le_rank_add_rank (s t : Submodule R M) :
    Module.rank R (s ⊔ t : Submodule R M) ≤ Module.rank R s + Module.rank R t := by
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    s t : Submodule R M
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Max.max s t) x)) (HAd …
  -/
  rw [← Submodule.rank_sup_add_rank_inf_eq]
  /-
    R : Type u_1
    M : Type u
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : HasRankNullity.{u, u_1} R
    s t : Submodule R M
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Max.max s t) x)) (HAd …
  -/
  exact self_le_add_right _ _
  /-
    🎉 no goals
  -/


/-- Given a family of `n` linearly independent vectors in a finite-dimensional space of
dimension `> n`, one may extend the family by another vector while retaining linear independence. -/
theorem exists_linearIndependent_snoc_of_lt_finrank {n : ℕ} {v : Fin n → M}
    (hv : LinearIndependent R v) (h : n < finrank R M) :
    ∃ (x : M), LinearIndependent R (Fin.snoc v x) :=
  exists_linearIndependent_snoc_of_lt_rank hv (lt_rank_of_lt_finrank h)


/-- Given a family of `n` linearly independent vectors in a finite-dimensional space of
dimension `> n`, one may extend the family by another vector while retaining linear independence. -/
theorem exists_linearIndependent_cons_of_lt_finrank {n : ℕ} {v : Fin n → M}
    (hv : LinearIndependent R v) (h : n < finrank R M) :
    ∃ (x : M), LinearIndependent R (Fin.cons x v) :=
  exists_linearIndependent_cons_of_lt_rank hv (lt_rank_of_lt_finrank h)


/-- Given a nonzero vector in a finite-dimensional space of dimension `> 1`, one may find another
vector linearly independent of the first one. -/
theorem exists_linearIndependent_pair_of_one_lt_finrank [NoZeroSMulDivisors R M]
    (h : 1 < finrank R M) {x : M} (hx : x ≠ 0) :
    ∃ y, LinearIndependent R ![x, y] :=
  exists_linearIndependent_pair_of_one_lt_rank (one_lt_rank_of_one_lt_finrank h) hx


/-- Rank-nullity theorem using `finrank`. -/
lemma Submodule.finrank_quotient_add_finrank [Module.Finite R M] (N : Submodule R M) :
    finrank R (M ⧸ N) + finrank R N = finrank R M := by
  rw [← Nat.cast_inj (R := Cardinal), Module.finrank_eq_rank, Nat.cast_add, Module.finrank_eq_rank,
    Submodule.finrank_eq_rank]
  /-
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    ⊢ Eq (HAdd.hAdd (Module.rank R (HasQuotient.Quotient M N)) (Module.rank R (Sub …
  -/
  exact HasRankNullity.rank_quotient_add_rank _
  /-
    🎉 no goals
  -/


/-- Rank-nullity theorem using `finrank` and subtraction. -/
lemma Submodule.finrank_quotient [Module.Finite R M] {S : Type*} [Ring S] [SMul R S] [Module S M]
    [IsScalarTower R S M] (N : Submodule S M) : finrank R (M ⧸ N) = finrank R M - finrank R N := by
  /-
    R : Type u_2
    M : Type u
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : HasRankNullity.{u, u_2} R
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Finite R M
    S : Type u_1
    inst✝³ : Ring S
    inst✝² : SMul R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    N : Submodule S M
    ⊢ Eq (Module.finrank R (HasQuotient.Quotient M N)) (HSub.hSub (Module.finrank  …
  -/
  rw [← (N.restrictScalars R).finrank_quotient_add_finrank]
  /-
    R : Type u_2
    M : Type u
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : HasRankNullity.{u, u_2} R
    inst✝⁵ : StrongRankCondition R
    inst✝⁴ : Module.Finite R M
    S : Type u_1
    inst✝³ : Ring S
    inst✝² : SMul R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    N : Submodule S M
    ⊢ Eq (Module.finrank R (HasQuotient.Quotient M N)) (HSub.hSub (HAdd.hAdd (Modu …
  -/
  exact Nat.eq_sub_of_add_eq rfl
  /-
    🎉 no goals
  -/


lemma Submodule.disjoint_ker_of_finrank_le [NoZeroSMulDivisors R M] {N : Type*} [AddCommGroup N]
    [Module R N] {L : Submodule R M} [Module.Finite R L] (f : M →ₗ[R] N)
    (h : finrank R L ≤ finrank R (L.map f)) :
    Disjoint L (LinearMap.ker f) := by
  refine disjoint_iff.mpr <| LinearMap.injective_domRestrict_iff.mp <| LinearMap.ker_eq_bot.mp <|
    Submodule.rank_eq_zero.mp ?_
  /-
    R : Type u_2
    M : Type u
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : HasRankNullity.{u, u_2} R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : NoZeroSMulDivisors R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    L : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id R) M N
    h : LE.le (Module.finrank R (Subtype fun x => Membership.mem L x)) (Module.fin …
    ⊢ Eq (Module.rank R (Subtype fun x => Membership.mem (LinearMap.ker (f.domRest …
  -/
  rw [← Submodule.finrank_eq_rank, Nat.cast_eq_zero]
  /-
    R : Type u_2
    M : Type u
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : HasRankNullity.{u, u_2} R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : NoZeroSMulDivisors R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    L : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id R) M N
    h : LE.le (Module.finrank R (Subtype fun x => Membership.mem L x)) (Module.fin …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.ker (f.domR …
  -/
  rw [← LinearMap.range_domRestrict] at h
  /-
    R : Type u_2
    M : Type u
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : HasRankNullity.{u, u_2} R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : NoZeroSMulDivisors R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    L : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id R) M N
    h : LE.le (Module.finrank R (Subtype fun x => Membership.mem L x)) (Module.fin …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.ker (f.domR …
  -/
  have := (LinearMap.ker (f.domRestrict L)).finrank_quotient_add_finrank
  /-
    R : Type u_2
    M : Type u
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : HasRankNullity.{u, u_2} R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : NoZeroSMulDivisors R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    L : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id R) M N
    h : LE.le (Module.finrank R (Subtype fun x => Membership.mem L x)) (Module.fin …
    this : Eq (HAdd.hAdd (Module.finrank R (HasQuotient.Quotient (Subtype fun x => …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.ker (f.domR …
  -/
  rw [LinearEquiv.finrank_eq (f.domRestrict L).quotKerEquivRange] at this
  /-
    R : Type u_2
    M : Type u
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : HasRankNullity.{u, u_2} R
    inst✝⁴ : StrongRankCondition R
    inst✝³ : NoZeroSMulDivisors R M
    N : Type u_1
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    L : Submodule R M
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id R) M N
    h : LE.le (Module.finrank R (Subtype fun x => Membership.mem L x)) (Module.fin …
    this : Eq (HAdd.hAdd (Module.finrank R (Subtype fun x => Membership.mem (Linea …
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.ker (f.domR …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma Submodule.exists_of_finrank_lt (N : Submodule R M) (h : finrank R N < finrank R M) :
    ∃ m : M, ∀ r : R, r ≠ 0 → r • m ∉ N := by
  obtain ⟨s, hs, hs'⟩ :=
    exists_finset_linearIndependent_of_le_finrank (R := R) (M := M ⧸ N) le_rfl
  obtain ⟨v, hv⟩ : s.Nonempty := by rwa [Finset.nonempty_iff_ne_empty, ne_eq, ← Finset.card_eq_zero,
    hs, finrank_quotient, tsub_eq_zero_iff_le, not_le]
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    h : LT.lt (Module.finrank R (Subtype fun x => Membership.mem N x)) (Module.fin …
    s : Finset (HasQuotient.Quotient M N)
    hs : Eq s.card (Module.finrank R (HasQuotient.Quotient M N))
    hs' : LinearIndependent R Subtype.val
    v : HasQuotient.Quotient M N
    hv : Membership.mem s v
    ⊢ Exists fun m => ∀ (r : R), Ne r 0 → Not (Membership.mem N (HSMul.hSMul r m))
  -/
  obtain ⟨v, rfl⟩ := N.mkQ_surjective v
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    h : LT.lt (Module.finrank R (Subtype fun x => Membership.mem N x)) (Module.fin …
    s : Finset (HasQuotient.Quotient M N)
    hs : Eq s.card (Module.finrank R (HasQuotient.Quotient M N))
    hs' : LinearIndependent R Subtype.val
    v : M
    hv : Membership.mem s (N.mkQ v)
    ⊢ Exists fun m => ∀ (r : R), Ne r 0 → Not (Membership.mem N (HSMul.hSMul r m))
  -/
  refine ⟨v, fun r hr ↦ mt ?_ hr⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : HasRankNullity.{u, u_1} R
    inst✝¹ : StrongRankCondition R
    inst✝ : Module.Finite R M
    N : Submodule R M
    h : LT.lt (Module.finrank R (Subtype fun x => Membership.mem N x)) (Module.fin …
    s : Finset (HasQuotient.Quotient M N)
    hs : Eq s.card (Module.finrank R (HasQuotient.Quotient M N))
    hs' : LinearIndependent R Subtype.val
    v : M
    hv : Membership.mem s (N.mkQ v)
    r : R
    hr : Ne r 0
    ⊢ Membership.mem N (HSMul.hSMul r v) → Eq r 0
  -/
  have := linearIndependent_iff.mp hs' (Finsupp.single ⟨_, hv⟩ r)
  rwa [Finsupp.linearCombination_single, Finsupp.single_eq_zero, ← LinearMap.map_smul,
    Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero] at this


