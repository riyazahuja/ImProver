/-- The torsion ideal of `x`, containing all `a` such that `a • x = 0`. -/
@[simps!]
def torsionOf (x : M) : Ideal R :=
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation on LinearMap.ker https://github.com/leanprover/lean4/issues/1629
  LinearMap.ker (LinearMap.toSpanSingleton R M x)


@[simp]
                                                         /-
                                                           R : Type u_1
                                                           M : Type u_2
                                                           inst✝² : Semiring R
                                                           inst✝¹ : AddCommMonoid M
                                                           inst✝ : Module R M
                                                           ⊢ Eq (Ideal.torsionOf R M 0) Top.top
                                                         -/
theorem torsionOf_zero : torsionOf R M (0 : M) = ⊤ := by simp [torsionOf]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem mem_torsionOf_iff (x : M) (a : R) : a ∈ torsionOf R M x ↔ a • x = 0 :=
  Iff.rfl


@[simp]
theorem torsionOf_eq_top_iff (m : M) : torsionOf R M m = ⊤ ↔ m = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    ⊢ Iff (Eq (Ideal.torsionOf R M m) Top.top) (Eq m 0)
  -/
  refine ⟨fun h => ?_, fun h => by simp [h]⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    h : Eq (Ideal.torsionOf R M m) Top.top
    ⊢ Eq m 0
  -/
  rw [← one_smul R m, ← mem_torsionOf_iff m (1 : R), h]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : M
    h : Eq (Ideal.torsionOf R M m) Top.top
    ⊢ Membership.mem Top.top 1
  -/
  exact Submodule.mem_top
  /-
    🎉 no goals
  -/


@[simp]
theorem torsionOf_eq_bot_iff_of_noZeroSMulDivisors [Nontrivial R] [NoZeroSMulDivisors R M] (m : M) :
    torsionOf R M m = ⊥ ↔ m ≠ 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroSMulDivisors R M
    m : M
    ⊢ Iff (Eq (Ideal.torsionOf R M m) Bot.bot) (Ne m 0)
  -/
  refine ⟨fun h contra => ?_, fun h => (Submodule.eq_bot_iff _).mpr fun r hr => ?_⟩
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroSMulDivisors R M
      m : M
      h : Eq (Ideal.torsionOf R M m) Bot.bot
      contra : Eq m 0
      ⊢ False
    -/
  · rw [contra, torsionOf_zero] at h
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroSMulDivisors R M
      m : M
      h : Eq Top.top Bot.bot
      contra : Eq m 0
      ⊢ False
    -/
    exact bot_ne_top.symm h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroSMulDivisors R M
      m : M
      h : Ne m 0
      r : R
      hr : Membership.mem (Ideal.torsionOf R M m) r
      ⊢ Eq r 0
    -/
  · rw [mem_torsionOf_iff, smul_eq_zero] at hr
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroSMulDivisors R M
      m : M
      h : Ne m 0
      r : R
      hr : Or (Eq r 0) (Eq m 0)
      ⊢ Eq r 0
    -/
    tauto
    /-
      🎉 no goals
    -/


/-- See also `iSupIndep.linearIndependent` which provides the same conclusion
but requires the stronger hypothesis `NoZeroSMulDivisors R M`. -/
theorem iSupIndep.linearIndependent' {ι R M : Type*} {v : ι → M} [Ring R]
    [AddCommGroup M] [Module R M] (hv : iSupIndep fun i => R ∙ v i)
    (h_ne_zero : ∀ i, Ideal.torsionOf R M (v i) = ⊥) : LinearIndependent R v := by
  /-
    ι : Type u_3
    R : Type u_4
    M : Type u_5
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : iSupIndep fun i => Submodule.span R (Singleton.singleton (v i))
    h_ne_zero : ∀ (i : ι), Eq (Ideal.torsionOf R M (v i)) Bot.bot
    ⊢ LinearIndependent R v
  -/
  refine linearIndependent_iff_not_smul_mem_span.mpr fun i r hi => ?_
  /-
    ι : Type u_3
    R : Type u_4
    M : Type u_5
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    hv : iSupIndep fun i => Submodule.span R (Singleton.singleton (v i))
    h_ne_zero : ∀ (i : ι), Eq (Ideal.torsionOf R M (v i)) Bot.bot
    i : ι
    r : R
    hi : Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Sing …
    ⊢ Eq r 0
  -/
  replace hv := iSupIndep_def.mp hv i
  /-
    ι : Type u_3
    R : Type u_4
    M : Type u_5
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h_ne_zero : ∀ (i : ι), Eq (Ideal.torsionOf R M (v i)) Bot.bot
    i : ι
    r : R
    hi : Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Sing …
    hv : Disjoint (Submodule.span R (Singleton.singleton (v i))) (iSup fun j => iS …
    ⊢ Eq r 0
  -/
  simp only [iSup_subtype', ← Submodule.span_range_eq_iSup (ι := Subtype _), disjoint_iff] at hv
  have : r • v i ∈ (⊥ : Submodule R M) := by
    rw [← hv, Submodule.mem_inf]
    refine ⟨Submodule.mem_span_singleton.mpr ⟨r, rfl⟩, ?_⟩
    convert hi
    ext
    simp
  /-
    ι : Type u_3
    R : Type u_4
    M : Type u_5
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h_ne_zero : ∀ (i : ι), Eq (Ideal.torsionOf R M (v i)) Bot.bot
    i : ι
    r : R
    hi : Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Sing …
    hv : Eq (Min.min (Submodule.span R (Singleton.singleton (v i))) (Submodule.spa …
    this : Membership.mem Bot.bot (HSMul.hSMul r (v i))
    ⊢ Eq r 0
  -/
  rw [← Submodule.mem_bot R, ← h_ne_zero i]
  /-
    ι : Type u_3
    R : Type u_4
    M : Type u_5
    v : ι → M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h_ne_zero : ∀ (i : ι), Eq (Ideal.torsionOf R M (v i)) Bot.bot
    i : ι
    r : R
    hi : Membership.mem (Submodule.span R (Set.image v (SDiff.sdiff Set.univ (Sing …
    hv : Eq (Min.min (Submodule.span R (Singleton.singleton (v i))) (Submodule.spa …
    this : Membership.mem Bot.bot (HSMul.hSMul r (v i))
    ⊢ Membership.mem (Ideal.torsionOf R M (v i)) r
  -/
  simpa using this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.linear_independent' := iSupIndep.linearIndependent'


/-- The span of `x` in `M` is isomorphic to `R` quotiented by the torsion ideal of `x`. -/
noncomputable def quotTorsionOfEquivSpanSingleton (x : M) : (R ⧸ torsionOf R M x) ≃ₗ[R] R ∙ x :=
  (LinearMap.toSpanSingleton R M x).quotKerEquivRange.trans <|
    LinearEquiv.ofEq _ _ (LinearMap.span_singleton_eq_range R M x).symm


@[simp]
theorem quotTorsionOfEquivSpanSingleton_apply_mk (x : M) (a : R) :
    quotTorsionOfEquivSpanSingleton R M x (Submodule.Quotient.mk a) =
      a • ⟨x, Submodule.mem_span_singleton_self x⟩ :=
  rfl


/-- The `a`-torsion submodule for `a` in `R`, containing all elements `x` of `M` such that
  `a • x = 0`. -/
@[simps!]
def torsionBy (a : R) : Submodule R M :=
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11036): broken dot notation on LinearMap.ker https://github.com/leanprover/lean4/issues/1629
  LinearMap.ker (DistribMulAction.toLinearMap R M a)


/-- The submodule containing all elements `x` of `M` such that `a • x = 0` for all `a` in `s`. -/
@[simps!]
def torsionBySet (s : Set R) : Submodule R M :=
  sInf (torsionBy R M '' s)

-- Porting note: torsion' had metavariables and factoring out this fixed it
-- perhaps there is a better fix

/-- The additive submonoid of all elements `x` of `M` such that `a • x = 0`
for some `a` in `S`. -/
@[simps!]
def torsion'AddSubMonoid (S : Type*) [CommMonoid S] [DistribMulAction S M] :
    AddSubmonoid M where
  carrier := { x | ∃ a : S, a • x = 0 }
  add_mem' := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_3
      inst✝¹ : CommMonoid S
      inst✝ : DistribMulAction S M
      ⊢ ∀ {a b : M}, Membership.mem (setOf fun x => Exists fun a => Eq (HSMul.hSMul  …
    -/
    intro x y ⟨a,hx⟩ ⟨b,hy⟩
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_3
      inst✝¹ : CommMonoid S
      inst✝ : DistribMulAction S M
      x y : M
      a : S
      hx : Eq (HSMul.hSMul a x) 0
      b : S
      hy : Eq (HSMul.hSMul b y) 0
      ⊢ Membership.mem (setOf fun x => Exists fun a => Eq (HSMul.hSMul a x) 0) (HAdd …
    -/
    use b * a
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_3
      inst✝¹ : CommMonoid S
      inst✝ : DistribMulAction S M
      x y : M
      a : S
      hx : Eq (HSMul.hSMul a x) 0
      b : S
      hy : Eq (HSMul.hSMul b y) 0
      ⊢ Eq (HSMul.hSMul (HMul.hMul b a) (HAdd.hAdd x y)) 0
    -/
    rw [smul_add, mul_smul, mul_comm, mul_smul, hx, hy, smul_zero, smul_zero, add_zero]
    /-
      🎉 no goals
    -/
  zero_mem' := ⟨1, smul_zero 1⟩


/-- The `S`-torsion submodule, containing all elements `x` of `M` such that `a • x = 0` for some
`a` in `S`. -/
@[simps!]
def torsion' (S : Type*) [CommMonoid S] [DistribMulAction S M] [SMulCommClass S R M] :
    Submodule R M :=
  { torsion'AddSubMonoid M S with
                                          /-
                                            R : Type u_1
                                            M : Type u_2
                                            inst✝⁵ : CommSemiring R
                                            inst✝⁴ : AddCommMonoid M
                                            inst✝³ : Module R M
                                            S : Type u_3
                                            inst✝² : CommMonoid S
                                            inst✝¹ : DistribMulAction S M
                                            inst✝ : SMulCommClass S R M
                                            a : R
                                            x : M
                                            x✝ : Membership.mem __src✝.carrier x
                                            b : S
                                            h : Eq (HSMul.hSMul b x) 0
                                            ⊢ Eq (HSMul.hSMul b (HSMul.hSMul a x)) 0
                                          -/
    smul_mem' := fun a x ⟨b, h⟩ => ⟨b, by rw [smul_comm, h, smul_zero]⟩}
                                          /-
                                            🎉 no goals
                                          -/


/-- The torsion submodule, containing all elements `x` of `M` such that `a • x = 0` for some
  non-zero-divisor `a` in `R`. -/
abbrev torsion :=
  torsion' R M R⁰


/-- An `a`-torsion module is a module where every element is `a`-torsion. -/
abbrev IsTorsionBy (a : R) :=
  ∀ ⦃x : M⦄, a • x = 0


/-- A module where every element is `a`-torsion for all `a` in `s`. -/
abbrev IsTorsionBySet (s : Set R) :=
  ∀ ⦃x : M⦄ ⦃a : s⦄, (a : R) • x = 0


/-- An `S`-torsion module is a module where every element is `a`-torsion for some `a` in `S`. -/
abbrev IsTorsion' (S : Type*) [SMul S M] :=
  ∀ ⦃x : M⦄, ∃ a : S, a • x = 0


/-- A torsion module is a module where every element is `a`-torsion for some non-zero-divisor `a`.
-/
abbrev IsTorsion :=
  ∀ ⦃x : M⦄, ∃ a : R⁰, a • x = 0


theorem isTorsionBySet_annihilator : IsTorsionBySet R M (Module.annihilator R M) :=
  fun _ r ↦ Module.mem_annihilator.mp r.2 _


lemma isSMulRegular_iff_torsionBy_eq_bot {R} (M : Type*)
    [CommRing R] [AddCommGroup M] [Module R M] (r : R) :
    IsSMulRegular M r ↔ Submodule.torsionBy R M r = ⊥ :=
  Iff.symm (DistribMulAction.toLinearMap R M r).ker_eq_bot


@[simp]
theorem smul_torsionBy (x : torsionBy R M a) : a • x = 0 :=
  Subtype.ext x.prop


@[simp]
theorem smul_coe_torsionBy (x : torsionBy R M a) : a • (x : M) = 0 :=
  x.prop


@[simp]
theorem mem_torsionBy_iff (x : M) : x ∈ torsionBy R M a ↔ a • x = 0 :=
  Iff.rfl


@[simp]
theorem mem_torsionBySet_iff (x : M) : x ∈ torsionBySet R M s ↔ ∀ a : s, (a : R) • x = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    x : M
    ⊢ Iff (Membership.mem (Submodule.torsionBySet R M s) x) (∀ (a : ↑s), Eq (HSMul …
  -/
  refine ⟨fun h ⟨a, ha⟩ => mem_sInf.mp h _ (Set.mem_image_of_mem _ ha), fun h => mem_sInf.mpr ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    x : M
    h : ∀ (a : ↑s), Eq (HSMul.hSMul (↑a) x) 0
    ⊢ ∀ (p : Submodule R M), Membership.mem (Set.image (Submodule.torsionBy R M) s …
  -/
  rintro _ ⟨a, ha, rfl⟩; exact h ⟨a, ha⟩
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem torsionBySet_singleton_eq : torsionBySet R M {a} = torsionBy R M a := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : R
    ⊢ Eq (Submodule.torsionBySet R M (Singleton.singleton a)) (Submodule.torsionBy …
  -/
  ext x
  simp only [mem_torsionBySet_iff, SetCoe.forall, Subtype.coe_mk, Set.mem_singleton_iff,
    forall_eq, mem_torsionBy_iff]


theorem torsionBySet_le_torsionBySet_of_subset {s t : Set R} (st : s ⊆ t) :
    torsionBySet R M t ≤ torsionBySet R M s :=
  sInf_le_sInf fun _ ⟨a, ha, h⟩ => ⟨a, st ha, h⟩


/-- Torsion by a set is torsion by the ideal generated by it. -/
theorem torsionBySet_eq_torsionBySet_span :
    torsionBySet R M s = torsionBySet R M (Ideal.span s) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    ⊢ Eq (Submodule.torsionBySet R M s) (Submodule.torsionBySet R M ↑(Ideal.span s))
  -/
  refine le_antisymm (fun x hx => ?_) (torsionBySet_le_torsionBySet_of_subset subset_span)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    x : M
    hx : Membership.mem (Submodule.torsionBySet R M s) x
    ⊢ Membership.mem (Submodule.torsionBySet R M ↑(Ideal.span s)) x
  -/
  rw [mem_torsionBySet_iff] at hx ⊢
  suffices Ideal.span s ≤ Ideal.torsionOf R M x by
    rintro ⟨a, ha⟩
    exact this ha
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    x : M
    hx : ∀ (a : ↑s), Eq (HSMul.hSMul (↑a) x) 0
    ⊢ LE.le (Ideal.span s) (Ideal.torsionOf R M x)
  -/
  rw [Ideal.span_le]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set R
    x : M
    hx : ∀ (a : ↑s), Eq (HSMul.hSMul (↑a) x) 0
    ⊢ HasSubset.Subset s ↑(Ideal.torsionOf R M x)
  -/
  exact fun a ha => hx ⟨a, ha⟩
  /-
    🎉 no goals
  -/


theorem torsionBySet_span_singleton_eq : torsionBySet R M (R ∙ a) = torsionBy R M a :=
  (torsionBySet_eq_torsionBySet_span _).symm.trans <| torsionBySet_singleton_eq _


theorem torsionBy_le_torsionBy_of_dvd (a b : R) (dvd : a ∣ b) :
    torsionBy R M a ≤ torsionBy R M b := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : R
    dvd : Dvd.dvd a b
    ⊢ LE.le (Submodule.torsionBy R M a) (Submodule.torsionBy R M b)
  -/
  rw [← torsionBySet_span_singleton_eq, ← torsionBySet_singleton_eq]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : R
    dvd : Dvd.dvd a b
    ⊢ LE.le (Submodule.torsionBySet R M ↑(Submodule.span R (Singleton.singleton a) …
  -/
  apply torsionBySet_le_torsionBySet_of_subset
  /-
    case st
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a b : R
    dvd : Dvd.dvd a b
    ⊢ HasSubset.Subset (Singleton.singleton b) ↑(Submodule.span R (Singleton.singl …
  -/
  rintro c (rfl : c = b); exact Ideal.mem_span_singleton.mpr dvd
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem torsionBy_one : torsionBy R M 1 = ⊥ :=
  eq_bot_iff.mpr fun _ h => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x✝ : M
      h : Membership.mem (Submodule.torsionBy R M 1) x✝
      ⊢ Membership.mem Bot.bot x✝
    -/
    rw [mem_torsionBy_iff, one_smul] at h
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x✝ : M
      h : Eq x✝ 0
      ⊢ Membership.mem Bot.bot x✝
    -/
    exact h
    /-
      🎉 no goals
    -/


@[simp]
theorem torsionBySet_univ : torsionBySet R M Set.univ = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (Submodule.torsionBySet R M Set.univ) Bot.bot
  -/
  rw [eq_bot_iff, ← torsionBy_one, ← torsionBySet_singleton_eq]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ LE.le (Submodule.torsionBySet R M Set.univ) (Submodule.torsionBySet R M (Sin …
  -/
  exact torsionBySet_le_torsionBySet_of_subset fun _ _ => trivial
  /-
    🎉 no goals
  -/


@[simp]
theorem isTorsionBySet_singleton_iff : IsTorsionBySet R M {a} ↔ IsTorsionBy R M a := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : R
    ⊢ Iff (Module.IsTorsionBySet R M (Singleton.singleton a)) (Module.IsTorsionBy  …
  -/
  refine ⟨fun h x => @h _ ⟨_, Set.mem_singleton _⟩, fun h x => ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : R
    h : Module.IsTorsionBy R M a
    x : M
    ⊢ ∀ ⦃a_1 : ↑(Singleton.singleton a)⦄, Eq (HSMul.hSMul (↑a_1) x) 0
  -/
  rintro ⟨b, rfl : b = a⟩; exact @h _
                           /-
                             🎉 no goals
                           -/


theorem isTorsionBySet_iff_torsionBySet_eq_top :
    IsTorsionBySet R M s ↔ Submodule.torsionBySet R M s = ⊤ :=
  ⟨fun h => eq_top_iff.mpr fun _ _ => (mem_torsionBySet_iff _ _).mpr <| @h _, fun h x => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set R
      h : Eq (Submodule.torsionBySet R M s) Top.top
      x : M
      ⊢ ∀ ⦃a : ↑s⦄, Eq (HSMul.hSMul (↑a) x) 0
    -/
    rw [← mem_torsionBySet_iff, h]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set R
      h : Eq (Submodule.torsionBySet R M s) Top.top
      x : M
      ⊢ Membership.mem Top.top x
    -/
    trivial⟩
    /-
      🎉 no goals
    -/


/-- An `a`-torsion module is a module whose `a`-torsion submodule is the full space. -/
theorem isTorsionBy_iff_torsionBy_eq_top : IsTorsionBy R M a ↔ torsionBy R M a = ⊤ := by
  rw [← torsionBySet_singleton_eq, ← isTorsionBySet_singleton_iff,
    isTorsionBySet_iff_torsionBySet_eq_top]


theorem isTorsionBySet_iff_is_torsion_by_span :
    IsTorsionBySet R M s ↔ IsTorsionBySet R M (Ideal.span s) := by
  rw [isTorsionBySet_iff_torsionBySet_eq_top, isTorsionBySet_iff_torsionBySet_eq_top,
    torsionBySet_eq_torsionBySet_span]


theorem isTorsionBySet_span_singleton_iff : IsTorsionBySet R M (R ∙ a) ↔ IsTorsionBy R M a :=
  (isTorsionBySet_iff_is_torsion_by_span _).symm.trans <| isTorsionBySet_singleton_iff _


theorem isTorsionBySet_iff_subseteq_ker_lsmul :
    IsTorsionBySet R M s ↔ s ⊆ LinearMap.ker (LinearMap.lsmul R M) where
  mp h r hr := LinearMap.mem_ker.mpr <| LinearMap.ext fun x => @h x ⟨r, hr⟩
  mpr | h, x, ⟨_, hr⟩ => DFunLike.congr_fun (LinearMap.mem_ker.mp (h hr)) x


theorem isTorsionBy_iff_mem_ker_lsmul :
    IsTorsionBy R M a ↔ a ∈ LinearMap.ker (LinearMap.lsmul R M) :=
  Iff.symm LinearMap.ext_iff


theorem torsionBySet_isTorsionBySet : IsTorsionBySet R (torsionBySet R M s) s :=
  fun ⟨_, hx⟩ a => Subtype.ext <| (mem_torsionBySet_iff _ _).mp hx a


/-- The `a`-torsion submodule is an `a`-torsion module. -/
theorem torsionBy_isTorsionBy : IsTorsionBy R (torsionBy R M a) a := smul_torsionBy a


@[simp]
theorem torsionBy_torsionBy_eq_top : torsionBy R (torsionBy R M a) a = ⊤ :=
  (isTorsionBy_iff_torsionBy_eq_top a).mp <| torsionBy_isTorsionBy a


@[simp]
theorem torsionBySet_torsionBySet_eq_top : torsionBySet R (torsionBySet R M s) s = ⊤ :=
  (isTorsionBySet_iff_torsionBySet_eq_top s).mp <| torsionBySet_isTorsionBySet s


theorem torsion_gc :
    @GaloisConnection (Submodule R M) (Ideal R)ᵒᵈ _ _ annihilator fun I =>
      torsionBySet R M ↑(OrderDual.ofDual I) :=
  fun _ _ =>
  ⟨fun h x hx => (mem_torsionBySet_iff _ _).mpr fun ⟨_, ha⟩ => mem_annihilator.mp (h ha) x hx,
    fun h a ha => mem_annihilator.mpr fun _ hx => (mem_torsionBySet_iff _ _).mp (h hx) ⟨a, ha⟩⟩


theorem iSup_torsionBySet_ideal_eq_torsionBySet_iInf
    (hp : (S : Set ι).Pairwise fun i j => p i ⊔ p j = ⊤) :
    ⨆ i ∈ S, torsionBySet R M (p i) = torsionBySet R M ↑(⨅ i ∈ S, p i) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    p : ι → Ideal R
    S : Finset ι
    hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
    ⊢ Eq (iSup fun i => iSup fun h => Submodule.torsionBySet R M ↑(p i)) (Submodul …
  -/
  rcases S.eq_empty_or_nonempty with h | h
    /-
      case inl
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : Eq S EmptyCollection.emptyCollection
      ⊢ Eq (iSup fun i => iSup fun h => Submodule.torsionBySet R M ↑(p i)) (Submodul …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    p : ι → Ideal R
    S : Finset ι
    hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
    h : S.Nonempty
    ⊢ Eq (iSup fun i => iSup fun h => Submodule.torsionBySet R M ↑(p i)) (Submodul …
  -/
  apply le_antisymm
    /-
      case inr.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      ⊢ LE.le (iSup fun i => iSup fun h => Submodule.torsionBySet R M ↑(p i)) (Submo …
    -/
  · apply iSup_le _
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      ⊢ ∀ (i : ι), LE.le (iSup fun h => Submodule.torsionBySet R M ↑(p i)) (Submodul …
    -/
    intro i
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      i : ι
      ⊢ LE.le (iSup fun h => Submodule.torsionBySet R M ↑(p i)) (Submodule.torsionBy …
    -/
    apply iSup_le _
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      i : ι
      ⊢ Membership.mem S i → LE.le (Submodule.torsionBySet R M ↑(p i)) (Submodule.to …
    -/
    intro is
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      i : ι
      is : Membership.mem S i
      ⊢ LE.le (Submodule.torsionBySet R M ↑(p i)) (Submodule.torsionBySet R M ↑(iInf …
    -/
    apply torsionBySet_le_torsionBySet_of_subset
    /-
      case st
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      i : ι
      is : Membership.mem S i
      ⊢ HasSubset.Subset ↑(iInf fun i => iInf fun h => p i) ↑(p i)
    -/
    exact (iInf_le (fun i => ⨅ _ : i ∈ S, p i) i).trans (iInf_le _ is)
    /-
      🎉 no goals
    -/
    /-
      case inr.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      ⊢ LE.le (Submodule.torsionBySet R M ↑(iInf fun i => iInf fun h => p i)) (iSup  …
    -/
  · intro x hx
    /-
      case inr.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      x : M
      hx : Membership.mem (Submodule.torsionBySet R M ↑(iInf fun i => iInf fun h =>  …
      ⊢ Membership.mem (iSup fun i => iSup fun h => Submodule.torsionBySet R M ↑(p i …
    -/
    rw [mem_iSup_finset_iff_exists_sum]
    obtain ⟨μ, hμ⟩ :=
      (mem_iSup_finset_iff_exists_sum _ _).mp
        ((Ideal.eq_top_iff_one _).mp <| (Ideal.iSup_iInf_eq_top_iff_pairwise h _).mpr hp)
    /-
      case inr.a.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      p : ι → Ideal R
      S : Finset ι
      hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
      h : S.Nonempty
      x : M
      hx : Membership.mem (Submodule.torsionBySet R M ↑(iInf fun i => iInf fun h =>  …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq (S.sum fun i => ↑(μ i)) 1
      ⊢ Exists fun μ => Eq (S.sum fun i => ↑(μ i)) x
    -/
    refine ⟨fun i => ⟨(μ i : R) • x, ?_⟩, ?_⟩
      /-
        case inr.a.intro.refine_1
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : Membership.mem (Submodule.torsionBySet R M ↑(iInf fun i => iInf fun h =>  …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        ⊢ Membership.mem (Submodule.torsionBySet R M ↑(p i)) (HSMul.hSMul (↑(μ i)) x)
      -/
    · rw [mem_torsionBySet_iff] at hx ⊢
      /-
        case inr.a.intro.refine_1
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        ⊢ ∀ (a : ↑↑(p i)), Eq (HSMul.hSMul (↑a) (HSMul.hSMul (↑(μ i)) x)) 0
      -/
      rintro ⟨a, ha⟩
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        ⊢ Eq (HSMul.hSMul (↑⟨a, ha⟩) (HSMul.hSMul (↑(μ i)) x)) 0
      -/
      rw [smul_smul]
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        ⊢ Eq (HSMul.hSMul (HMul.hMul ↑⟨a, ha⟩ ↑(μ i)) x) 0
      -/
      suffices a * μ i ∈ ⨅ i ∈ S, p i from hx ⟨_, this⟩
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        ⊢ Membership.mem (iInf fun i => iInf fun h => p i) (HMul.hMul a ↑(μ i))
      -/
      rw [mem_iInf]
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        ⊢ ∀ (i_1 : ι), Membership.mem (iInf fun h => p i_1) (HMul.hMul a ↑(μ i))
      -/
      intro j
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        j : ι
        ⊢ Membership.mem (iInf fun h => p j) (HMul.hMul a ↑(μ i))
      -/
      rw [mem_iInf]
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        j : ι
        ⊢ Membership.mem S j → Membership.mem (p j) (HMul.hMul a ↑(μ i))
      -/
      intro hj
      /-
        case inr.a.intro.refine_1.mk
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        i : ι
        a : R
        ha : Membership.mem (↑(p i)) a
        j : ι
        hj : Membership.mem S j
        ⊢ Membership.mem (p j) (HMul.hMul a ↑(μ i))
      -/
      by_cases ij : j = i
        /-
          case pos
          R : Type u_1
          M : Type u_2
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          ι : Type u_3
          p : ι → Ideal R
          S : Finset ι
          hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
          h : S.Nonempty
          x : M
          hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (S.sum fun i => ↑(μ i)) 1
          i : ι
          a : R
          ha : Membership.mem (↑(p i)) a
          j : ι
          hj : Membership.mem S j
          ij : Eq j i
          ⊢ Membership.mem (p j) (HMul.hMul a ↑(μ i))
        -/
      · rw [ij]
        /-
          case pos
          R : Type u_1
          M : Type u_2
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          ι : Type u_3
          p : ι → Ideal R
          S : Finset ι
          hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
          h : S.Nonempty
          x : M
          hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (S.sum fun i => ↑(μ i)) 1
          i : ι
          a : R
          ha : Membership.mem (↑(p i)) a
          j : ι
          hj : Membership.mem S j
          ij : Eq j i
          ⊢ Membership.mem (p i) (HMul.hMul a ↑(μ i))
        -/
        exact Ideal.mul_mem_right _ _ ha
        /-
          🎉 no goals
        -/
        /-
          case neg
          R : Type u_1
          M : Type u_2
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          ι : Type u_3
          p : ι → Ideal R
          S : Finset ι
          hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
          h : S.Nonempty
          x : M
          hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (S.sum fun i => ↑(μ i)) 1
          i : ι
          a : R
          ha : Membership.mem (↑(p i)) a
          j : ι
          hj : Membership.mem S j
          ij : Not (Eq j i)
          ⊢ Membership.mem (p j) (HMul.hMul a ↑(μ i))
        -/
      · have := coe_mem (μ i)
        /-
          case neg
          R : Type u_1
          M : Type u_2
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          ι : Type u_3
          p : ι → Ideal R
          S : Finset ι
          hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
          h : S.Nonempty
          x : M
          hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (S.sum fun i => ↑(μ i)) 1
          i : ι
          a : R
          ha : Membership.mem (↑(p i)) a
          j : ι
          hj : Membership.mem S j
          ij : Not (Eq j i)
          this : Membership.mem (iInf fun j => iInf fun x => iInf fun x => p j) ↑(μ i)
          ⊢ Membership.mem (p j) (HMul.hMul a ↑(μ i))
        -/
        simp only [mem_iInf] at this
        /-
          case neg
          R : Type u_1
          M : Type u_2
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          ι : Type u_3
          p : ι → Ideal R
          S : Finset ι
          hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
          h : S.Nonempty
          x : M
          hx : ∀ (a : ↑↑(iInf fun i => iInf fun h => p i)), Eq (HSMul.hSMul (↑a) x) 0
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (S.sum fun i => ↑(μ i)) 1
          i : ι
          a : R
          ha : Membership.mem (↑(p i)) a
          j : ι
          hj : Membership.mem S j
          ij : Not (Eq j i)
          this : ∀ (i_1 : ι), Membership.mem S i_1 → Ne i_1 i → Membership.mem (p i_1) ↑ …
          ⊢ Membership.mem (p j) (HMul.hMul a ↑(μ i))
        -/
        exact Ideal.mul_mem_left _ _ (this j hj ij)
        /-
          🎉 no goals
        -/
      /-
        case inr.a.intro.refine_2
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Type u_3
        p : ι → Ideal R
        S : Finset ι
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        h : S.Nonempty
        x : M
        hx : Membership.mem (Submodule.torsionBySet R M ↑(iInf fun i => iInf fun h =>  …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (S.sum fun i => ↑(μ i)) 1
        ⊢ Eq (S.sum fun i => ↑((fun i => ⟨HSMul.hSMul (↑(μ i)) x, ⋯⟩) i)) x
      -/
    · rw [← Finset.sum_smul, hμ, one_smul]
      /-
        🎉 no goals
      -/

-- Porting note: iSup_torsionBySet_ideal_eq_torsionBySet_iInf now requires DecidableEq ι

theorem supIndep_torsionBySet_ideal (hp : (S : Set ι).Pairwise fun i j => p i ⊔ p j = ⊤) :
    S.SupIndep fun i => torsionBySet R M <| p i :=
  fun T hT i hi hiT => by
  rw [disjoint_iff, Finset.sup_eq_iSup,
    iSup_torsionBySet_ideal_eq_torsionBySet_iInf fun i hi j hj ij => hp (hT hi) (hT hj) ij]
  have := GaloisConnection.u_inf
    (b₁ := OrderDual.toDual (p i)) (b₂ := OrderDual.toDual (⨅ i ∈ T, p i)) (torsion_gc R M)
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    p : ι → Ideal R
    S : Finset ι
    hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
    T : Finset ι
    hT : HasSubset.Subset T S
    i : ι
    hi : Membership.mem S i
    hiT : Not (Membership.mem T i)
    this : Eq (Submodule.torsionBySet R M ↑(OrderDual.ofDual (Min.min (OrderDual.t …
    ⊢ Eq (Min.min ((fun i => Submodule.torsionBySet R M ↑(p i)) i) (Submodule.tors …
  -/
  dsimp at this ⊢
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    p : ι → Ideal R
    S : Finset ι
    hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
    T : Finset ι
    hT : HasSubset.Subset T S
    i : ι
    hi : Membership.mem S i
    hiT : Not (Membership.mem T i)
    this : Eq (Submodule.torsionBySet R M ↑(Max.max (p i) (iInf fun i => iInf fun  …
    ⊢ Eq (Min.min (Submodule.torsionBySet R M ↑(p i)) (Submodule.torsionBySet R M  …
  -/
  rw [← this, Ideal.sup_iInf_eq_top, top_coe, torsionBySet_univ]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    p : ι → Ideal R
    S : Finset ι
    hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
    T : Finset ι
    hT : HasSubset.Subset T S
    i : ι
    hi : Membership.mem S i
    hiT : Not (Membership.mem T i)
    this : Eq (Submodule.torsionBySet R M ↑(Max.max (p i) (iInf fun i => iInf fun  …
    ⊢ ∀ (i_1 : ι), Membership.mem T i_1 → Eq (Max.max (p i) (p i_1)) Top.top
  -/
  intro j hj; apply hp hi (hT hj); rintro rfl; exact hiT hj
                                               /-
                                                 🎉 no goals
                                               -/


theorem iSup_torsionBy_eq_torsionBy_prod (hq : (S : Set ι).Pairwise <| (IsCoprime on q)) :
    ⨆ i ∈ S, torsionBy R M (q i) = torsionBy R M (∏ i ∈ S, q i) := by
  rw [← torsionBySet_span_singleton_eq, Ideal.submodule_span_eq, ←
    Ideal.finset_inf_span_singleton _ _ hq, Finset.inf_eq_iInf, ←
    iSup_torsionBySet_ideal_eq_torsionBySet_iInf]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      S : Finset ι
      q : ι → R
      hq : (↑S).Pairwise (Function.onFun IsCoprime q)
      ⊢ Eq (iSup fun i => iSup fun h => Submodule.torsionBy R M (q i)) (iSup fun i = …
    -/
  · congr
    /-
      case e_s
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      S : Finset ι
      q : ι → R
      hq : (↑S).Pairwise (Function.onFun IsCoprime q)
      ⊢ Eq (fun i => iSup fun h => Submodule.torsionBy R M (q i)) fun i => iSup fun  …
    -/
    ext : 1
    /-
      case e_s.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      S : Finset ι
      q : ι → R
      hq : (↑S).Pairwise (Function.onFun IsCoprime q)
      x✝ : ι
      ⊢ Eq (iSup fun h => Submodule.torsionBy R M (q x✝)) (iSup fun h => Submodule.t …
    -/
    congr
    /-
      case e_s.h.e_s
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      S : Finset ι
      q : ι → R
      hq : (↑S).Pairwise (Function.onFun IsCoprime q)
      x✝ : ι
      ⊢ Eq (fun h => Submodule.torsionBy R M (q x✝)) fun h => Submodule.torsionBySet …
    -/
    ext : 1
    /-
      case e_s.h.e_s.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Type u_3
      S : Finset ι
      q : ι → R
      hq : (↑S).Pairwise (Function.onFun IsCoprime q)
      x✝¹ : ι
      x✝ : Membership.mem S x✝¹
      ⊢ Eq (Submodule.torsionBy R M (q x✝¹)) (Submodule.torsionBySet R M ↑(Ideal.spa …
    -/
    exact (torsionBySet_span_singleton_eq _).symm
    /-
      🎉 no goals
    -/
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    S : Finset ι
    q : ι → R
    hq : (↑S).Pairwise (Function.onFun IsCoprime q)
    ⊢ (↑S).Pairwise fun i j => Eq (Max.max (Ideal.span (Singleton.singleton (q i)) …
  -/
  exact fun i hi j hj ij => (Ideal.sup_eq_top_iff_isCoprime _ _).mpr (hq hi hj ij)
  /-
    🎉 no goals
  -/


theorem supIndep_torsionBy (hq : (S : Set ι).Pairwise <| (IsCoprime on q)) :
    S.SupIndep fun i => torsionBy R M <| q i := by
  convert supIndep_torsionBySet_ideal (M := M) fun i hi j hj ij =>
      (Ideal.sup_eq_top_iff_isCoprime (q i) _).mpr <| hq hi hj ij
  /-
    case h.e'_6.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_3
    S : Finset ι
    q : ι → R
    hq : (↑S).Pairwise (Function.onFun IsCoprime q)
    x✝ : ι
    ⊢ Eq (Submodule.torsionBy R M (q x✝)) (Submodule.torsionBySet R M ↑(Ideal.span …
  -/
  exact (torsionBySet_span_singleton_eq (R := R) (M := M) _).symm
  /-
    🎉 no goals
  -/


/-- If the `p i` are pairwise coprime, a `⨅ i, p i`-torsion module is the internal direct sum of
its `p i`-torsion submodules. -/
theorem torsionBySet_isInternal {p : ι → Ideal R}
    (hp : (S : Set ι).Pairwise fun i j => p i ⊔ p j = ⊤)
    (hM : Module.IsTorsionBySet R M (⨅ i ∈ S, p i : Ideal R)) :
    DirectSum.IsInternal fun i : S => torsionBySet R M <| p i :=
  DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top
    (iSupIndep_iff_supIndep.mpr <| supIndep_torsionBySet_ideal hp)
    (by
      /-
        R : Type u_1
        M : Type u_2
        inst✝³ : CommRing R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        ι : Type u_3
        inst✝ : DecidableEq ι
        S : Finset ι
        p : ι → Ideal R
        hp : (↑S).Pairwise fun i j => Eq (Max.max (p i) (p j)) Top.top
        hM : Module.IsTorsionBySet R M ↑(iInf fun i => iInf fun h => p i)
        ⊢ Eq (iSup fun i => Submodule.torsionBySet R M ↑(p ↑i)) Top.top
      -/
      apply (iSup_subtype'' ↑S fun i => torsionBySet R M <| p i).trans
      -- Porting note: times out if we change apply below to <|
      apply (iSup_torsionBySet_ideal_eq_torsionBySet_iInf hp).trans <|
        (Module.isTorsionBySet_iff_torsionBySet_eq_top _).mp hM)


/-- If the `q i` are pairwise coprime, a `∏ i, q i`-torsion module is the internal direct sum of
its `q i`-torsion submodules. -/
theorem torsionBy_isInternal {q : ι → R} (hq : (S : Set ι).Pairwise <| (IsCoprime on q))
    (hM : Module.IsTorsionBy R M <| ∏ i ∈ S, q i) :
    DirectSum.IsInternal fun i : S => torsionBy R M <| q i := by
  rw [← Module.isTorsionBySet_span_singleton_iff, Ideal.submodule_span_eq, ←
    Ideal.finset_inf_span_singleton _ _ hq, Finset.inf_eq_iInf] at hM
  convert torsionBySet_isInternal
      (fun i hi j hj ij => (Ideal.sup_eq_top_iff_isCoprime (q i) _).mpr <| hq hi hj ij) hM
  /-
    case h.e'_8.h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : DecidableEq ι
    S : Finset ι
    q : ι → R
    hq : (↑S).Pairwise (Function.onFun IsCoprime q)
    hM : Module.IsTorsionBySet R M ↑(iInf fun a => iInf fun h => Ideal.span (Singl …
    x✝ : Subtype fun x => Membership.mem S x
    ⊢ Eq (Submodule.torsionBy R M (q ↑x✝)) (Submodule.torsionBySet R M ↑(Ideal.spa …
  -/
  exact (torsionBySet_span_singleton_eq _ (R := R) (M := M)).symm
  /-
    🎉 no goals
  -/


/-- can't be an instance because `hM` can't be inferred -/
def IsTorsionBySet.hasSMul (hM : IsTorsionBySet R M I) : SMul (R ⧸ I) M where
  smul b x := I.liftQ (LinearMap.lsmul R M)
                ((isTorsionBySet_iff_subseteq_ker_lsmul _).mp hM) b x


/-- can't be an instance because `hM` can't be inferred -/
abbrev IsTorsionBy.hasSMul (hM : IsTorsionBy R M r) : SMul (R ⧸ Ideal.span {r}) M :=
  ((isTorsionBySet_span_singleton_iff r).mpr hM).hasSMul


@[simp]
theorem IsTorsionBySet.mk_smul (hM : IsTorsionBySet R M I) (b : R) (x : M) :
    haveI := hM.hasSMul
    Ideal.Quotient.mk I b • x = b • x :=
  rfl


@[simp]
theorem IsTorsionBy.mk_smul (hM : IsTorsionBy R M r) (b : R) (x : M) :
    haveI := hM.hasSMul
    Ideal.Quotient.mk (Ideal.span {r}) b • x = b • x :=
  rfl


/-- An `(R ⧸ I)`-module is an `R`-module which `IsTorsionBySet R M I`. -/
def IsTorsionBySet.module (hM : IsTorsionBySet R M I) : Module (R ⧸ I) M :=
  letI := hM.hasSMul; I.mkQ_surjective.moduleLeft _ (IsTorsionBySet.mk_smul hM)


instance IsTorsionBySet.isScalarTower (hM : IsTorsionBySet R M I)
    {S : Type*} [SMul S R] [SMul S M] [IsScalarTower S R M] [IsScalarTower S R R] :
    @IsScalarTower S (R ⧸ I) M _ (IsTorsionBySet.module hM).toSMul _ :=
  -- Porting note: still needed to be fed the Module R / I M instance
  @IsScalarTower.mk S (R ⧸ I) M _ (IsTorsionBySet.module hM).toSMul _
    (fun b d x => Quotient.inductionOn' d fun c => (smul_assoc b c x : _))


/-- An `(R ⧸ Ideal.span {r})`-module is an `R`-module for which `IsTorsionBy R M r`. -/
abbrev IsTorsionBy.module (hM : IsTorsionBy R M r) : Module (R ⧸ Ideal.span {r}) M :=
  ((isTorsionBySet_span_singleton_iff r).mpr hM).module


/-- Any module is also a module over the quotient of the ring by the annihilator.
Not an instance because it causes synthesis failures / timeouts. -/
def quotientAnnihilator : Module (R ⧸ Module.annihilator R M) M :=
  (isTorsionBySet_annihilator R M).module


theorem isTorsionBy_quotient_iff (N : Submodule R M) (r : R) :
    IsTorsionBy R (M⧸N) r ↔ ∀ x, r • x ∈ N :=
  Iff.trans N.mkQ_surjective.forall <| forall_congr' fun _ =>
    Submodule.Quotient.mk_eq_zero N


theorem IsTorsionBy.quotient (N : Submodule R M) {r : R}
    (h : IsTorsionBy R M r) : IsTorsionBy R (M⧸N) r :=
  (isTorsionBy_quotient_iff N r).mpr fun x => @h x ▸ N.zero_mem


theorem isTorsionBySet_quotient_iff (N : Submodule R M) (s : Set R) :
    IsTorsionBySet R (M⧸N) s ↔ ∀ x, ∀ r ∈ s, r • x ∈ N :=
  Iff.trans N.mkQ_surjective.forall <| forall_congr' fun _ =>
    Iff.trans Subtype.forall <| forall₂_congr fun _ _ =>
      Submodule.Quotient.mk_eq_zero N


theorem IsTorsionBySet.quotient (N : Submodule R M) {s}
    (h : IsTorsionBySet R M s) : IsTorsionBySet R (M⧸N) s :=
  (isTorsionBySet_quotient_iff N s).mpr fun x r h' => @h x ⟨r, h'⟩ ▸ N.zero_mem


lemma isTorsionBySet_quotient_set_smul :
    IsTorsionBySet R (M⧸s • (⊤ : Submodule R M)) s :=
  (isTorsionBySet_quotient_iff _ _).mpr fun _ _ h =>
    mem_set_smul_of_mem_mem h mem_top


lemma isTorsionBy_quotient_element_smul :
    IsTorsionBy R (M⧸r • (⊤ : Submodule R M)) r :=
  (isTorsionBy_quotient_iff _ _).mpr (smul_mem_pointwise_smul · r ⊤ ⟨⟩)


lemma isTorsionBySet_quotient_ideal_smul :
    IsTorsionBySet R (M⧸I • (⊤ : Submodule R M)) I :=
  (isTorsionBySet_quotient_iff _ _).mpr fun _ _ h => smul_mem_smul h ⟨⟩


instance : Module (R ⧸ Ideal.span s) (M ⧸ s • (⊤ : Submodule R M)) :=
  ((isTorsionBySet_iff_is_torsion_by_span s).mp
    (isTorsionBySet_quotient_set_smul M s)).module


instance : Module (R ⧸ I) (M ⧸ I • (⊤ : Submodule R M)) :=
  (isTorsionBySet_quotient_ideal_smul M I).module


instance : Module (R ⧸ Ideal.span {r}) (M ⧸ r • (⊤ : Submodule R M)) :=
  (isTorsionBy_quotient_element_smul M r).module


lemma Quotient.mk_smul_mk (r : R) (m : M) :
    Ideal.Quotient.mk I r •
      Submodule.Quotient.mk (p := (I • ⊤ : Submodule R M)) m =
      Submodule.Quotient.mk (p := (I • ⊤ : Submodule R M)) (r • m) :=
  rfl


instance (I : Ideal R) : Module (R ⧸ I) (torsionBySet R M I) :=
  -- Porting note: times out without the (R := R)
  Module.IsTorsionBySet.module <| torsionBySet_isTorsionBySet (R := R) I


@[simp]
theorem torsionBySet.mk_smul (I : Ideal R) (b : R) (x : torsionBySet R M I) :
    Ideal.Quotient.mk I b • x = b • x :=
  rfl


instance (I : Ideal R) {S : Type*} [SMul S R] [SMul S M] [IsScalarTower S R M]
    [IsScalarTower S R R] : IsScalarTower S (R ⧸ I) (torsionBySet R M I) :=
  inferInstance


/-- The `a`-torsion submodule as an `(R ⧸ R∙a)`-module. -/
instance instModuleQuotientTorsionBy (a : R) : Module (R ⧸ R ∙ a) (torsionBy R M a) :=
  Module.IsTorsionBySet.module <|
    (Module.isTorsionBySet_span_singleton_iff a).mpr <| torsionBy_isTorsionBy a

-- Porting note: added for torsionBy.mk_ideal_smul

instance (a : R) : Module (R ⧸ Ideal.span {a}) (torsionBy R M a) :=
   inferInstanceAs <| Module (R ⧸ R ∙ a) (torsionBy R M a)

-- Porting note: added because torsionBy.mk_smul simplifies

@[simp]
theorem torsionBy.mk_ideal_smul (a b : R) (x : torsionBy R M a) :
    (Ideal.Quotient.mk (Ideal.span {a})) b • x = b • x :=
  rfl


theorem torsionBy.mk_smul (a b : R) (x : torsionBy R M a) :
    Ideal.Quotient.mk (R ∙ a) b • x = b • x :=
  rfl


instance (a : R) {S : Type*} [SMul S R] [SMul S M] [IsScalarTower S R M] [IsScalarTower S R R] :
    IsScalarTower S (R ⧸ R ∙ a) (torsionBy R M a) :=
  inferInstance


/-- Given an `R`-module `M` and an element `a` in `R`, submodules of the `a`-torsion submodule of
`M` do not depend on whether we take scalars to be `R` or `R ⧸ R ∙ a`. -/
def submodule_torsionBy_orderIso (a : R) :
    Submodule (R ⧸ R ∙ a) (torsionBy R M a) ≃o Submodule R (torsionBy R M a) :=
  { restrictScalarsEmbedding R (R ⧸ R ∙ a) (torsionBy R M a) with
    invFun := fun p ↦
      { carrier := p
        add_mem' := add_mem
        zero_mem' := p.zero_mem
                        /-
                          R : Type u_1
                          M : Type u_2
                          inst✝² : CommRing R
                          inst✝¹ : AddCommGroup M
                          inst✝ : Module R M
                          a : R
                          p : Submodule R (Subtype fun x => Membership.mem (Submodule.torsionBy R M a) x)
                          ⊢ ∀ (c : HasQuotient.Quotient R (Submodule.span R (Singleton.singleton a))) {x …
                        -/
        smul_mem' := by rintro ⟨b⟩; exact p.smul_mem b }
                                    /-
                                      🎉 no goals
                                    -/
                   /-
                     R : Type u_1
                     M : Type u_2
                     inst✝² : CommRing R
                     inst✝¹ : AddCommGroup M
                     inst✝ : Module R M
                     a : R
                     ⊢ Function.LeftInverse (fun p => { carrier := ↑p, add_mem' := ⋯, zero_mem' :=  …
                   -/
    left_inv := by intro; ext; simp [restrictScalarsEmbedding]
                               /-
                                 🎉 no goals
                               -/
                    /-
                      R : Type u_1
                      M : Type u_2
                      inst✝² : CommRing R
                      inst✝¹ : AddCommGroup M
                      inst✝ : Module R M
                      a : R
                      ⊢ Function.RightInverse (fun p => { carrier := ↑p, add_mem' := ⋯, zero_mem' := …
                    -/
    right_inv := by intro; ext; simp [restrictScalarsEmbedding] }
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem mem_torsion'_iff (x : M) : x ∈ torsion' R M S ↔ ∃ a : S, a • x = 0 :=
  Iff.rfl


theorem mem_torsion_iff (x : M) : x ∈ torsion R M ↔ ∃ a : R⁰, a • x = 0 :=
  Iff.rfl


@[simps]
instance : SMul S (torsion' R M S) :=
  ⟨fun s x =>
    ⟨s • (x : M), by
      /-
        R : Type u_1
        M : Type u_2
        inst✝⁵ : CommSemiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_3
        inst✝² : CommMonoid S
        inst✝¹ : DistribMulAction S M
        inst✝ : SMulCommClass S R M
        s : S
        x : Subtype fun x => Membership.mem (Submodule.torsion' R M S) x
        ⊢ Membership.mem (Submodule.torsion' R M S) (HSMul.hSMul s ↑x)
      -/
      obtain ⟨x, a, h⟩ := x
      /-
        case mk.intro
        R : Type u_1
        M : Type u_2
        inst✝⁵ : CommSemiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_3
        inst✝² : CommMonoid S
        inst✝¹ : DistribMulAction S M
        inst✝ : SMulCommClass S R M
        s : S
        x : M
        a : S
        h : Eq (HSMul.hSMul a x) 0
        ⊢ Membership.mem (Submodule.torsion' R M S) (HSMul.hSMul s ↑⟨x, ⋯⟩)
      -/
      use a
      /-
        case h
        R : Type u_1
        M : Type u_2
        inst✝⁵ : CommSemiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_3
        inst✝² : CommMonoid S
        inst✝¹ : DistribMulAction S M
        inst✝ : SMulCommClass S R M
        s : S
        x : M
        a : S
        h : Eq (HSMul.hSMul a x) 0
        ⊢ Eq (HSMul.hSMul a (HSMul.hSMul s ↑⟨x, ⋯⟩)) 0
      -/
      dsimp
      /-
        case h
        R : Type u_1
        M : Type u_2
        inst✝⁵ : CommSemiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_3
        inst✝² : CommMonoid S
        inst✝¹ : DistribMulAction S M
        inst✝ : SMulCommClass S R M
        s : S
        x : M
        a : S
        h : Eq (HSMul.hSMul a x) 0
        ⊢ Eq (HSMul.hSMul a (HSMul.hSMul s x)) 0
      -/
      rw [smul_comm, h, smul_zero]⟩⟩
      /-
        🎉 no goals
      -/


instance : DistribMulAction S (torsion' R M S) :=
  Subtype.coe_injective.distribMulAction (torsion' R M S).subtype.toAddMonoidHom fun (_ : S) _ =>
    rfl


instance : SMulCommClass S R (torsion' R M S) :=
  ⟨fun _ _ _ => Subtype.ext <| smul_comm _ _ _⟩


/-- An `S`-torsion module is a module whose `S`-torsion submodule is the full space. -/
theorem isTorsion'_iff_torsion'_eq_top : IsTorsion' M S ↔ torsion' R M S = ⊤ :=
  ⟨fun h => eq_top_iff.mpr fun _ _ => @h _, fun h x => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁵ : CommSemiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_3
      inst✝² : CommMonoid S
      inst✝¹ : DistribMulAction S M
      inst✝ : SMulCommClass S R M
      h : Eq (Submodule.torsion' R M S) Top.top
      x : M
      ⊢ Exists fun a => Eq (HSMul.hSMul a x) 0
    -/
    rw [← @mem_torsion'_iff R, h]
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁵ : CommSemiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_3
      inst✝² : CommMonoid S
      inst✝¹ : DistribMulAction S M
      inst✝ : SMulCommClass S R M
      h : Eq (Submodule.torsion' R M S) Top.top
      x : M
      ⊢ Membership.mem Top.top x
    -/
    trivial⟩
    /-
      🎉 no goals
    -/


/-- The `S`-torsion submodule is an `S`-torsion module. -/
theorem torsion'_isTorsion' : IsTorsion' (torsion' R M S) S := fun ⟨_, ⟨a, h⟩⟩ => ⟨a, Subtype.ext h⟩


@[simp]
theorem torsion'_torsion'_eq_top : torsion' R (torsion' R M S) S = ⊤ :=
  (isTorsion'_iff_torsion'_eq_top S).mp <| torsion'_isTorsion' S


/-- The torsion submodule of the torsion submodule (viewed as a module) is the full
torsion module. -/
theorem torsion_torsion_eq_top : torsion R (torsion R M) = ⊤ :=
  torsion'_torsion'_eq_top R⁰


/-- The torsion submodule is always a torsion module. -/
theorem torsion_isTorsion : Module.IsTorsion R (torsion R M) :=
  torsion'_isTorsion' R⁰


theorem _root_.Module.isTorsionBySet_annihilator_top :
    Module.IsTorsionBySet R M (⊤ : Submodule R M).annihilator := fun x ha =>
  mem_annihilator.mp ha.prop x mem_top


theorem _root_.Submodule.annihilator_top_inter_nonZeroDivisors [Module.Finite R M]
    (hM : Module.IsTorsion R M) : ((⊤ : Submodule R M).annihilator : Set R) ∩ R⁰ ≠ ∅ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    ⊢ Ne (Inter.inter ↑Top.top.annihilator ↑(nonZeroDivisors R)) EmptyCollection.e …
  -/
  obtain ⟨S, hS⟩ := ‹Module.Finite R M›.out
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    ⊢ Ne (Inter.inter ↑Top.top.annihilator ↑(nonZeroDivisors R)) EmptyCollection.e …
  -/
  refine Set.Nonempty.ne_empty ⟨_, ?_, (∏ x ∈ S, (@hM x).choose : R⁰).prop⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    ⊢ Membership.mem ↑Top.top.annihilator ↑(S.prod fun x => ⋯.choose)
  -/
  rw [Submonoid.coe_finset_prod, SetLike.mem_coe, ← hS, mem_annihilator_span]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    ⊢ ∀ (n : ↑↑S), Eq (HSMul.hSMul (S.prod fun i => ↑⋯.choose) ↑n) 0
  -/
  intro n
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    hM : Module.IsTorsion R M
    S : Finset M
    hS : Eq (Submodule.span R ↑S) Top.top
    n : ↑↑S
    ⊢ Eq (HSMul.hSMul (S.prod fun i => ↑⋯.choose) ↑n) 0
  -/
  letI := Classical.decEq M
  rw [← Finset.prod_erase_mul _ _ n.prop, mul_smul, ← Submonoid.smul_def, (@hM n).choose_spec,
    smul_zero]


theorem coe_torsion_eq_annihilator_ne_bot :
    (torsion R M : Set M) = { x : M | (R ∙ x).annihilator ≠ ⊥ } := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    ⊢ Eq (↑(Submodule.torsion R M)) (setOf fun x => Ne (Submodule.span R (Singleto …
  -/
  ext x; simp_rw [Submodule.ne_bot_iff, mem_annihilator, mem_span_singleton]
  exact
    ⟨fun ⟨a, hax⟩ =>
      ⟨a, fun _ ⟨b, hb⟩ => by rw [← hb, smul_comm, ← Submonoid.smul_def, hax, smul_zero],
        nonZeroDivisors.coe_ne_zero _⟩,
      fun ⟨a, hax, ha⟩ => ⟨⟨_, mem_nonZeroDivisors_of_ne_zero ha⟩, hax x ⟨1, one_smul _ _⟩⟩⟩


/-- A module over a domain has `NoZeroSMulDivisors` iff its torsion submodule is trivial. -/
theorem noZeroSMulDivisors_iff_torsion_eq_bot : NoZeroSMulDivisors R M ↔ torsion R M = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    ⊢ Iff (NoZeroSMulDivisors R M) (Eq (Submodule.torsion R M) Bot.bot)
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      h : NoZeroSMulDivisors R M
      ⊢ Eq (Submodule.torsion R M) Bot.bot
    -/
  · haveI : NoZeroSMulDivisors R M := h
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      h this : NoZeroSMulDivisors R M
      ⊢ Eq (Submodule.torsion R M) Bot.bot
    -/
    rw [eq_bot_iff]
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      h this : NoZeroSMulDivisors R M
      ⊢ LE.le (Submodule.torsion R M) Bot.bot
    -/
    rintro x ⟨a, hax⟩
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      h this : NoZeroSMulDivisors R M
      x : M
      a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      hax : Eq (HSMul.hSMul a x) 0
      ⊢ Membership.mem Bot.bot x
    -/
    change (a : R) • x = 0 at hax
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : NoZeroDivisors R
      inst✝ : Nontrivial R
      h this : NoZeroSMulDivisors R M
      x : M
      a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
      hax : Eq (HSMul.hSMul (↑a) x) 0
      ⊢ Membership.mem Bot.bot x
    -/
    cases' eq_zero_or_eq_zero_of_smul_eq_zero hax with h0 h0
      /-
        case mp.intro.inl
        R : Type u_1
        M : Type u_2
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        h this : NoZeroSMulDivisors R M
        x : M
        a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hax : Eq (HSMul.hSMul (↑a) x) 0
        h0 : Eq (↑a) 0
        ⊢ Membership.mem Bot.bot x
      -/
    · exfalso
      /-
        case mp.intro.inl
        R : Type u_1
        M : Type u_2
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        h this : NoZeroSMulDivisors R M
        x : M
        a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hax : Eq (HSMul.hSMul (↑a) x) 0
        h0 : Eq (↑a) 0
        ⊢ False
      -/
      exact nonZeroDivisors.coe_ne_zero a h0
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.inr
        R : Type u_1
        M : Type u_2
        inst✝⁴ : CommSemiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : NoZeroDivisors R
        inst✝ : Nontrivial R
        h this : NoZeroSMulDivisors R M
        x : M
        a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hax : Eq (HSMul.hSMul (↑a) x) 0
        h0 : Eq x 0
        ⊢ Membership.mem Bot.bot x
      -/
    · exact h0
      /-
        🎉 no goals
      -/
  · exact
      { eq_zero_or_eq_zero_of_smul_eq_zero := fun {a} {x} hax => by
          by_cases ha : a = 0
          · left
            exact ha
          · right
            rw [← mem_bot R, ← h]
            exact ⟨⟨a, mem_nonZeroDivisors_of_ne_zero ha⟩, hax⟩ }


lemma torsion_int {G} [AddCommGroup G] :
    (torsion ℤ G).toAddSubgroup = AddCommGroup.torsion G := by
  /-
    G : Type u_3
    inst✝ : AddCommGroup G
    ⊢ Eq (Submodule.torsion Int G).toAddSubgroup (AddCommGroup.torsion G)
  -/
  ext x
  /-
    case h
    G : Type u_3
    inst✝ : AddCommGroup G
    x : G
    ⊢ Iff (Membership.mem (Submodule.torsion Int G).toAddSubgroup x) (Membership.m …
  -/
  refine ((isOfFinAddOrder_iff_zsmul_eq_zero (x := x)).trans ?_).symm
  /-
    case h
    G : Type u_3
    inst✝ : AddCommGroup G
    x : G
    ⊢ Iff (Exists fun n => And (Ne n 0) (Eq (HSMul.hSMul n x) 0)) (Membership.mem  …
  -/
  simp [mem_nonZeroDivisors_iff_ne_zero]
  /-
    🎉 no goals
  -/


/-- Quotienting by the torsion submodule gives a torsion-free module. -/
@[simp]
theorem torsion_eq_bot : torsion R (M ⧸ torsion R M) = ⊥ :=
  eq_bot_iff.mpr fun z =>
    Quotient.inductionOn' z fun x ⟨a, hax⟩ => by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        z : HasQuotient.Quotient M (Submodule.torsion R M)
        x : M
        x✝ : Membership.mem (Submodule.torsion R (HasQuotient.Quotient M (Submodule.to …
        a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hax : Eq (HSMul.hSMul a (Quotient.mk'' x)) 0
        ⊢ Membership.mem Bot.bot (Quotient.mk'' x)
      -/
      rw [Quotient.mk''_eq_mk, ← Quotient.mk_smul, Quotient.mk_eq_zero] at hax
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        z : HasQuotient.Quotient M (Submodule.torsion R M)
        x : M
        x✝ : Membership.mem (Submodule.torsion R (HasQuotient.Quotient M (Submodule.to …
        a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hax : Membership.mem (Submodule.torsion R M) (HSMul.hSMul a x)
        ⊢ Membership.mem Bot.bot (Quotient.mk'' x)
      -/
      rw [mem_bot, Quotient.mk''_eq_mk, Quotient.mk_eq_zero]
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        z : HasQuotient.Quotient M (Submodule.torsion R M)
        x : M
        x✝ : Membership.mem (Submodule.torsion R (HasQuotient.Quotient M (Submodule.to …
        a : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hax : Membership.mem (Submodule.torsion R M) (HSMul.hSMul a x)
        ⊢ Membership.mem (Submodule.torsion R M) x
      -/
      cases' hax with b h
      /-
        case intro
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        z : HasQuotient.Quotient M (Submodule.torsion R M)
        x : M
        x✝ : Membership.mem (Submodule.torsion R (HasQuotient.Quotient M (Submodule.to …
        a b : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        h : Eq (HSMul.hSMul b (HSMul.hSMul a x)) 0
        ⊢ Membership.mem (Submodule.torsion R M) x
      -/
      exact ⟨b * a, (mul_smul _ _ _).trans h⟩
      /-
        🎉 no goals
      -/


instance noZeroSMulDivisors [IsDomain R] : NoZeroSMulDivisors R (M ⧸ torsion R M) :=
  noZeroSMulDivisors_iff_torsion_eq_bot.mpr torsion_eq_bot


theorem isTorsion'_powers_iff (p : R) :
    IsTorsion' M (Submonoid.powers p) ↔ ∀ x : M, ∃ n : ℕ, p ^ n • x = 0 := by
  -- Porting note: previous term proof was having trouble elaborating
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Monoid R
    inst✝¹ : AddCommMonoid M
    inst✝ : DistribMulAction R M
    p : R
    ⊢ Iff (Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      ⊢ Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) x) …
    -/
  · intro h x
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) x)
      x : M
      ⊢ Exists fun n => Eq (HSMul.hSMul (HPow.hPow p n) x) 0
    -/
    let ⟨⟨a, ⟨n, hn⟩⟩, hx⟩ := @h x
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) x)
      x : M
      a : R
      n : Nat
      hn : Eq ((fun x => HPow.hPow p x) n) a
      hx : Eq (HSMul.hSMul ⟨a, ⋯⟩ x) 0
      ⊢ Exists fun n => Eq (HSMul.hSMul (HPow.hPow p n) x) 0
    -/
    dsimp at hn
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) x)
      x : M
      a : R
      n : Nat
      hn : Eq (HPow.hPow p n) a
      hx : Eq (HSMul.hSMul ⟨a, ⋯⟩ x) 0
      ⊢ Exists fun n => Eq (HSMul.hSMul (HPow.hPow p n) x) 0
    -/
    use n
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) x)
      x : M
      a : R
      n : Nat
      hn : Eq (HPow.hPow p n) a
      hx : Eq (HSMul.hSMul ⟨a, ⋯⟩ x) 0
      ⊢ Eq (HSMul.hSMul (HPow.hPow p n) x) 0
    -/
    rw [hn]
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) x)
      x : M
      a : R
      n : Nat
      hn : Eq (HPow.hPow p n) a
      hx : Eq (HSMul.hSMul ⟨a, ⋯⟩ x) 0
      ⊢ Eq (HSMul.hSMul a x) 0
    -/
    apply hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      ⊢ (∀ (x : M), Exists fun n => Eq (HSMul.hSMul (HPow.hPow p n) x) 0) → Module.I …
    -/
  · intro h x
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : ∀ (x : M), Exists fun n => Eq (HSMul.hSMul (HPow.hPow p n) x) 0
      x : M
      ⊢ Exists fun a => Eq (HSMul.hSMul a x) 0
    -/
    let ⟨n, hn⟩ := h x
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Monoid R
      inst✝¹ : AddCommMonoid M
      inst✝ : DistribMulAction R M
      p : R
      h : ∀ (x : M), Exists fun n => Eq (HSMul.hSMul (HPow.hPow p n) x) 0
      x : M
      n : Nat
      hn : Eq (HSMul.hSMul (HPow.hPow p n) x) 0
      ⊢ Exists fun a => Eq (HSMul.hSMul a x) 0
    -/
    exact ⟨⟨_, ⟨n, rfl⟩⟩, hn⟩
    /-
      🎉 no goals
    -/


/-- In a `p ^ ∞`-torsion module (that is, a module where all elements are cancelled by scalar
multiplication by some power of `p`), the smallest `n` such that `p ^ n • x = 0`. -/
def pOrder {p : R} (hM : IsTorsion' M <| Submonoid.powers p) (x : M)
    [∀ n : ℕ, Decidable (p ^ n • x = 0)] :=
  Nat.find <| (isTorsion'_powers_iff p).mp hM x


@[simp]
theorem pow_pOrder_smul {p : R} (hM : IsTorsion' M <| Submonoid.powers p) (x : M)
    [∀ n : ℕ, Decidable (p ^ n • x = 0)] : p ^ pOrder hM x • x = 0 :=
  Nat.find_spec <| (isTorsion'_powers_iff p).mp hM x


theorem exists_isTorsionBy {p : R} (hM : IsTorsion' M <| Submonoid.powers p) (d : ℕ) (hd : d ≠ 0)
    (s : Fin d → M) (hs : span R (Set.range s) = ⊤) :
    ∃ j : Fin d, Module.IsTorsionBy R M (p ^ pOrder hM (s j)) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : (x : M) → Decidable (Eq x 0)
    p : R
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    d : Nat
    hd : Ne d 0
    s : Fin d → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    ⊢ Exists fun j => Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM (s  …
  -/
  let oj := List.argmax (fun i => pOrder hM <| s i) (List.finRange d)
  have hoj : oj.isSome :=
    Option.ne_none_iff_isSome.mp fun eq_none =>
      hd <| List.finRange_eq_nil.mp <| List.argmax_eq_none.mp eq_none
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : (x : M) → Decidable (Eq x 0)
    p : R
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    d : Nat
    hd : Ne d 0
    s : Fin d → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    oj : Option (Fin d) := List.argmax (fun i => Submodule.pOrder hM (s i)) (List. …
    hoj : Eq oj.isSome Bool.true
    ⊢ Exists fun j => Module.IsTorsionBy R M (HPow.hPow p (Submodule.pOrder hM (s  …
  -/
  use Option.get _ hoj
  rw [isTorsionBy_iff_torsionBy_eq_top, eq_top_iff, ← hs, Submodule.span_le,
    Set.range_subset_iff]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : (x : M) → Decidable (Eq x 0)
    p : R
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    d : Nat
    hd : Ne d 0
    s : Fin d → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    oj : Option (Fin d) := List.argmax (fun i => Submodule.pOrder hM (s i)) (List. …
    hoj : Eq oj.isSome Bool.true
    ⊢ ∀ (y : Fin d), Membership.mem (↑(Submodule.torsionBy R M (HPow.hPow p (Submo …
  -/
  intro i; change (p ^ pOrder hM (s (Option.get oj hoj))) • s i = 0
  have : pOrder hM (s i) ≤ pOrder hM (s <| Option.get _ hoj) :=
    List.le_of_mem_argmax (List.mem_finRange i) (Option.get_mem hoj)
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : (x : M) → Decidable (Eq x 0)
    p : R
    hM : Module.IsTorsion' M (Subtype fun x => Membership.mem (Submonoid.powers p) …
    d : Nat
    hd : Ne d 0
    s : Fin d → M
    hs : Eq (Submodule.span R (Set.range s)) Top.top
    oj : Option (Fin d) := List.argmax (fun i => Submodule.pOrder hM (s i)) (List. …
    hoj : Eq oj.isSome Bool.true
    i : Fin d
    this : LE.le (Submodule.pOrder hM (s i)) (Submodule.pOrder hM (s (oj.get hoj)))
    ⊢ Eq (HSMul.hSMul (HPow.hPow p (Submodule.pOrder hM (s (oj.get hoj)))) (s i)) 0
  -/
  rw [← Nat.sub_add_cancel this, pow_add, mul_smul, pow_pOrder_smul, smul_zero]
  /-
    🎉 no goals
  -/


theorem torsionBy_eq_span_singleton {R : Type w} [CommRing R] (a b : R) (ha : a ∈ R⁰) :
    torsionBy R (R ⧸ R ∙ a * b) a = R ∙ mk (R ∙ a * b) b := by
  /-
    R : Type w
    inst✝ : CommRing R
    a b : R
    ha : Membership.mem (nonZeroDivisors R) a
    ⊢ Eq (Submodule.torsionBy R (HasQuotient.Quotient R (Submodule.span R (Singlet …
  -/
  ext x; rw [mem_torsionBy_iff, Submodule.mem_span_singleton]
  /-
    case h
    R : Type w
    inst✝ : CommRing R
    a b : R
    ha : Membership.mem (nonZeroDivisors R) a
    x : HasQuotient.Quotient R (Submodule.span R (Singleton.singleton (HMul.hMul a …
    ⊢ Iff (Eq (HSMul.hSMul a x) 0) (Exists fun a_1 => Eq (HSMul.hSMul a_1 ((Ideal. …
  -/
  obtain ⟨x, rfl⟩ := mk_surjective x; constructor <;> intro h
    /-
      case h.intro.mp
      R : Type w
      inst✝ : CommRing R
      a b : R
      ha : Membership.mem (nonZeroDivisors R) a
      x : R
      h : Eq (HSMul.hSMul a ((Ideal.Quotient.mk (Submodule.span R (Singleton.singlet …
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ((Ideal.Quotient.mk (Submodule.span R  …
    -/
  · rw [← mk_eq_mk, ← Quotient.mk_smul, Quotient.mk_eq_zero, Submodule.mem_span_singleton] at h
    /-
      case h.intro.mp
      R : Type w
      inst✝ : CommRing R
      a b : R
      ha : Membership.mem (nonZeroDivisors R) a
      x : R
      h : Exists fun a_1 => Eq (HSMul.hSMul a_1 (HMul.hMul a b)) (HSMul.hSMul a x)
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ((Ideal.Quotient.mk (Submodule.span R  …
    -/
    obtain ⟨c, h⟩ := h
    rw [smul_eq_mul, smul_eq_mul, mul_comm, mul_assoc, mul_cancel_left_mem_nonZeroDivisors ha,
      mul_comm] at h
    /-
      case h.intro.mp.intro
      R : Type w
      inst✝ : CommRing R
      a b : R
      ha : Membership.mem (nonZeroDivisors R) a
      x c : R
      h : Eq (HMul.hMul c b) x
      ⊢ Exists fun a_1 => Eq (HSMul.hSMul a_1 ((Ideal.Quotient.mk (Submodule.span R  …
    -/
    use c
    /-
      case h
      R : Type w
      inst✝ : CommRing R
      a b : R
      ha : Membership.mem (nonZeroDivisors R) a
      x c : R
      h : Eq (HMul.hMul c b) x
      ⊢ Eq (HSMul.hSMul c ((Ideal.Quotient.mk (Submodule.span R (Singleton.singleton …
    -/
    rw [← h, ← mk_eq_mk, ← Quotient.mk_smul, smul_eq_mul, mk_eq_mk]
    /-
      🎉 no goals
    -/
    /-
      case h.intro.mpr
      R : Type w
      inst✝ : CommRing R
      a b : R
      ha : Membership.mem (nonZeroDivisors R) a
      x : R
      h : Exists fun a_1 => Eq (HSMul.hSMul a_1 ((Ideal.Quotient.mk (Submodule.span  …
      ⊢ Eq (HSMul.hSMul a ((Ideal.Quotient.mk (Submodule.span R (Singleton.singleton …
    -/
  · obtain ⟨c, h⟩ := h
    rw [← h, smul_comm, ← mk_eq_mk, ← Quotient.mk_smul,
      (Quotient.mk_eq_zero _).mpr <| mem_span_singleton_self _, smul_zero]


theorem isTorsion_iff_isTorsion_nat [AddCommMonoid M] :
    AddMonoid.IsTorsion M ↔ Module.IsTorsion ℕ M := by
  /-
    M : Type u_2
    inst✝ : AddCommMonoid M
    ⊢ Iff (AddMonoid.IsTorsion M) (Module.IsTorsion Nat M)
  -/
  refine ⟨fun h x => ?_, fun h x => ?_⟩
    /-
      case refine_1
      M : Type u_2
      inst✝ : AddCommMonoid M
      h : AddMonoid.IsTorsion M
      x : M
      ⊢ Exists fun a => Eq (HSMul.hSMul a x) 0
    -/
  · obtain ⟨n, h0, hn⟩ := (h x).exists_nsmul_eq_zero
    /-
      case refine_1.intro.intro
      M : Type u_2
      inst✝ : AddCommMonoid M
      h : AddMonoid.IsTorsion M
      x : M
      n : Nat
      h0 : LT.lt 0 n
      hn : Eq (HSMul.hSMul n x) 0
      ⊢ Exists fun a => Eq (HSMul.hSMul a x) 0
    -/
    exact ⟨⟨n, mem_nonZeroDivisors_of_ne_zero <| ne_of_gt h0⟩, hn⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_2
      inst✝ : AddCommMonoid M
      h : Module.IsTorsion Nat M
      x : M
      ⊢ IsOfFinAddOrder x
    -/
  · rw [isOfFinAddOrder_iff_nsmul_eq_zero]
    /-
      case refine_2
      M : Type u_2
      inst✝ : AddCommMonoid M
      h : Module.IsTorsion Nat M
      x : M
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HSMul.hSMul n x) 0)
    -/
    obtain ⟨n, hn⟩ := @h x
    /-
      case refine_2.intro
      M : Type u_2
      inst✝ : AddCommMonoid M
      h : Module.IsTorsion Nat M
      x : M
      n : Subtype fun x => Membership.mem (nonZeroDivisors Nat) x
      hn : Eq (HSMul.hSMul n x) 0
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HSMul.hSMul n x) 0)
    -/
    exact ⟨n, Nat.pos_of_ne_zero (nonZeroDivisors.coe_ne_zero _), hn⟩
    /-
      🎉 no goals
    -/


theorem isTorsion_iff_isTorsion_int [AddCommGroup M] :
    AddMonoid.IsTorsion M ↔ Module.IsTorsion ℤ M := by
  /-
    M : Type u_2
    inst✝ : AddCommGroup M
    ⊢ Iff (AddMonoid.IsTorsion M) (Module.IsTorsion Int M)
  -/
  refine ⟨fun h x => ?_, fun h x => ?_⟩
    /-
      case refine_1
      M : Type u_2
      inst✝ : AddCommGroup M
      h : AddMonoid.IsTorsion M
      x : M
      ⊢ Exists fun a => Eq (HSMul.hSMul a x) 0
    -/
  · obtain ⟨n, h0, hn⟩ := (h x).exists_nsmul_eq_zero
    exact
      ⟨⟨n, mem_nonZeroDivisors_of_ne_zero <| ne_of_gt <| Int.natCast_pos.mpr h0⟩,
        (natCast_zsmul _ _).trans hn⟩
    /-
      case refine_2
      M : Type u_2
      inst✝ : AddCommGroup M
      h : Module.IsTorsion Int M
      x : M
      ⊢ IsOfFinAddOrder x
    -/
  · rw [isOfFinAddOrder_iff_nsmul_eq_zero]
    /-
      case refine_2
      M : Type u_2
      inst✝ : AddCommGroup M
      h : Module.IsTorsion Int M
      x : M
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HSMul.hSMul n x) 0)
    -/
    obtain ⟨n, hn⟩ := @h x
    /-
      case refine_2.intro
      M : Type u_2
      inst✝ : AddCommGroup M
      h : Module.IsTorsion Int M
      x : M
      n : Subtype fun x => Membership.mem (nonZeroDivisors Int) x
      hn : Eq (HSMul.hSMul n x) 0
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HSMul.hSMul n x) 0)
    -/
    exact ⟨_, Int.natAbs_pos.2 (nonZeroDivisors.coe_ne_zero n), natAbs_nsmul_eq_zero.2 hn⟩
    /-
      🎉 no goals
    -/


/-- The additive `n`-torsion subgroup for an integer `n`. -/
@[reducible]
def torsionBy : AddSubgroup A :=
  (Submodule.torsionBy ℤ A n).toAddSubgroup


@[inherit_doc]
scoped notation:max (priority := high) A"["n"]" => torsionBy A n


lemma torsionBy.neg : A[-n] = A[n] := by
  /-
    A : Type u_3
    inst✝ : AddCommGroup A
    n : Int
    ⊢ Eq (AddSubgroup.torsionBy A (Neg.neg n)) (AddSubgroup.torsionBy A n)
  -/
  ext a
  /-
    case h
    A : Type u_3
    inst✝ : AddCommGroup A
    n : Int
    a : A
    ⊢ Iff (Membership.mem (AddSubgroup.torsionBy A (Neg.neg n)) a) (Membership.mem …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma torsionBy.nsmul (x : A[n]) : n • x = 0 :=
  Nat.cast_smul_eq_nsmul ℤ n x ▸ Submodule.smul_torsionBy ..


lemma torsionBy.nsmul_iff {x : A} :
    x ∈ A[n] ↔ n • x = 0 :=
  Nat.cast_smul_eq_nsmul ℤ n x ▸ Submodule.mem_torsionBy_iff ..


lemma torsionBy.mod_self_nsmul (s : ℕ) (x : A[n])  :
    s • x = (s % n) • x :=
  nsmul_eq_mod_nsmul s (torsionBy.nsmul x)


lemma torsionBy.mod_self_nsmul' (s : ℕ) {x : A} (h : x ∈ A[n]) :
    s • x = (s % n) • x :=
  nsmul_eq_mod_nsmul s (torsionBy.nsmul_iff.mp h)


/-- For a natural number `n`, the `n`-torsion subgroup of `A` is a `ZMod n` module. -/
def torsionBy.zmodModule : Module (ZMod n) A[n] :=
  AddCommGroup.zmodModule torsionBy.nsmul


@[simp]
lemma infinite_range_add_smul_iff
    [AddCommGroup M] [Ring R] [Module R M] [Infinite R] [NoZeroSMulDivisors R M] (x y : M) :
    (Set.range <| fun r : R ↦ x + r • y).Infinite ↔ y ≠ 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Ring R
    inst✝² : Module R M
    inst✝¹ : Infinite R
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    ⊢ Iff (Set.range fun r => HAdd.hAdd x (HSMul.hSMul r y)).Infinite (Ne y 0)
  -/
  refine ⟨fun h hy ↦ by simp [hy] at h, fun h ↦ Set.infinite_range_of_injective fun r s hrs ↦ ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Ring R
    inst✝² : Module R M
    inst✝¹ : Infinite R
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    h : Ne y 0
    r s : R
    hrs : Eq (HAdd.hAdd x (HSMul.hSMul r y)) (HAdd.hAdd x (HSMul.hSMul s y))
    ⊢ Eq r s
  -/
  rw [add_right_inj] at hrs
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Ring R
    inst✝² : Module R M
    inst✝¹ : Infinite R
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    h : Ne y 0
    r s : R
    hrs : Eq (HSMul.hSMul r y) (HSMul.hSMul s y)
    ⊢ Eq r s
  -/
  exact smul_left_injective _ h hrs
  /-
    🎉 no goals
  -/


@[simp]
lemma infinite_range_add_nsmul_iff [AddCommGroup M] [NoZeroSMulDivisors ℤ M] (x y : M) :
    (Set.range <| fun n : ℕ ↦ x + n • y).Infinite ↔ y ≠ 0 := by
  /-
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : NoZeroSMulDivisors Int M
    x y : M
    ⊢ Iff (Set.range fun n => HAdd.hAdd x (HSMul.hSMul n y)).Infinite (Ne y 0)
  -/
  refine ⟨fun h hy ↦ by simp [hy] at h, fun h ↦ Set.infinite_range_of_injective fun r s hrs ↦ ?_⟩
  /-
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : NoZeroSMulDivisors Int M
    x y : M
    h : Ne y 0
    r s : Nat
    hrs : Eq (HAdd.hAdd x (HSMul.hSMul r y)) (HAdd.hAdd x (HSMul.hSMul s y))
    ⊢ Eq r s
  -/
  rw [add_right_inj, ← natCast_zsmul, ← natCast_zsmul] at hrs
  /-
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : NoZeroSMulDivisors Int M
    x y : M
    h : Ne y 0
    r s : Nat
    hrs : Eq (HSMul.hSMul (↑r) y) (HSMul.hSMul (↑s) y)
    ⊢ Eq r s
  -/
  simpa using smul_left_injective _ h hrs
  /-
    🎉 no goals
  -/


