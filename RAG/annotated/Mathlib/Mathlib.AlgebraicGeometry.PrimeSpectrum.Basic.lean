/-- The Zariski topology on the prime spectrum of a commutative (semi)ring is defined
via the closed sets of the topology: they are exactly those sets that are the zero locus
of a subset of the ring. -/
instance zariskiTopology : TopologicalSpace (PrimeSpectrum R) :=
                                                                              /-
                                                                                R : Type u
                                                                                S : Type v
                                                                                inst✝¹ : CommSemiring R
                                                                                inst✝ : CommSemiring S
                                                                                ⊢ Eq (PrimeSpectrum.zeroLocus Set.univ) EmptyCollection.emptyCollection
                                                                              -/
  TopologicalSpace.ofClosed (Set.range PrimeSpectrum.zeroLocus) ⟨Set.univ, by simp⟩
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    (by
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        ⊢ ∀ (A : Set (Set (PrimeSpectrum R))), HasSubset.Subset A (Set.range PrimeSpec …
      -/
      intro Zs h
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        Zs : Set (Set (PrimeSpectrum R))
        h : HasSubset.Subset Zs (Set.range PrimeSpectrum.zeroLocus)
        ⊢ Membership.mem (Set.range PrimeSpectrum.zeroLocus) Zs.sInter
      -/
      rw [Set.sInter_eq_iInter]
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        Zs : Set (Set (PrimeSpectrum R))
        h : HasSubset.Subset Zs (Set.range PrimeSpectrum.zeroLocus)
        ⊢ Membership.mem (Set.range PrimeSpectrum.zeroLocus) (Set.iInter fun i => ↑i)
      -/
      choose f hf using fun i : Zs => h i.prop
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        Zs : Set (Set (PrimeSpectrum R))
        h : HasSubset.Subset Zs (Set.range PrimeSpectrum.zeroLocus)
        f : ↑Zs → Set R
        hf : ∀ (i : ↑Zs), Eq (PrimeSpectrum.zeroLocus (f i)) ↑i
        ⊢ Membership.mem (Set.range PrimeSpectrum.zeroLocus) (Set.iInter fun i => ↑i)
      -/
      simp only [← hf]
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        Zs : Set (Set (PrimeSpectrum R))
        h : HasSubset.Subset Zs (Set.range PrimeSpectrum.zeroLocus)
        f : ↑Zs → Set R
        hf : ∀ (i : ↑Zs), Eq (PrimeSpectrum.zeroLocus (f i)) ↑i
        ⊢ Membership.mem (Set.range PrimeSpectrum.zeroLocus) (Set.iInter fun i => Prim …
      -/
      exact ⟨_, zeroLocus_iUnion _⟩)
      /-
        🎉 no goals
      -/
    (by
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        ⊢ ∀ (A : Set (PrimeSpectrum R)), Membership.mem (Set.range PrimeSpectrum.zeroL …
      -/
      rintro _ ⟨s, rfl⟩ _ ⟨t, rfl⟩
      /-
        case intro.intro
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        s t : Set R
        ⊢ Membership.mem (Set.range PrimeSpectrum.zeroLocus) (Union.union (PrimeSpectr …
      -/
      exact ⟨_, (union_zeroLocus s t).symm⟩)
      /-
        🎉 no goals
      -/


theorem isOpen_iff (U : Set (PrimeSpectrum R)) : IsOpen U ↔ ∃ s, Uᶜ = zeroLocus s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    U : Set (PrimeSpectrum R)
    ⊢ Iff (IsOpen U) (Exists fun s => Eq (HasCompl.compl U) (PrimeSpectrum.zeroLoc …
  -/
  simp only [@eq_comm _ Uᶜ]; rfl
                             /-
                               🎉 no goals
                             -/


theorem isClosed_iff_zeroLocus (Z : Set (PrimeSpectrum R)) : IsClosed Z ↔ ∃ s, Z = zeroLocus s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    Z : Set (PrimeSpectrum R)
    ⊢ Iff (IsClosed Z) (Exists fun s => Eq Z (PrimeSpectrum.zeroLocus s))
  -/
  rw [← isOpen_compl_iff, isOpen_iff, compl_compl]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_zeroLocus_ideal (Z : Set (PrimeSpectrum R)) :
    IsClosed Z ↔ ∃ I : Ideal R, Z = zeroLocus I :=
  (isClosed_iff_zeroLocus _).trans
    ⟨fun ⟨s, hs⟩ => ⟨_, (zeroLocus_span s).substr hs⟩, fun ⟨I, hI⟩ => ⟨I, hI⟩⟩


theorem isClosed_iff_zeroLocus_radical_ideal (Z : Set (PrimeSpectrum R)) :
    IsClosed Z ↔ ∃ I : Ideal R, I.IsRadical ∧ Z = zeroLocus I :=
  (isClosed_iff_zeroLocus_ideal _).trans
    ⟨fun ⟨I, hI⟩ => ⟨_, I.radical_isRadical, (zeroLocus_radical I).substr hI⟩, fun ⟨I, _, hI⟩ =>
      ⟨I, hI⟩⟩


theorem isClosed_zeroLocus (s : Set R) : IsClosed (zeroLocus s) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ IsClosed (PrimeSpectrum.zeroLocus s)
  -/
  rw [isClosed_iff_zeroLocus]
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Exists fun s_1 => Eq (PrimeSpectrum.zeroLocus s) (PrimeSpectrum.zeroLocus s_1)
  -/
  exact ⟨s, rfl⟩
  /-
    🎉 no goals
  -/


theorem zeroLocus_vanishingIdeal_eq_closure (t : Set (PrimeSpectrum R)) :
    zeroLocus (vanishingIdeal t : Set R) = closure t := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    ⊢ Eq (PrimeSpectrum.zeroLocus ↑(PrimeSpectrum.vanishingIdeal t)) (closure t)
  -/
  rcases isClosed_iff_zeroLocus (closure t) |>.mp isClosed_closure with ⟨I, hI⟩
  rw [subset_antisymm_iff, (isClosed_zeroLocus _).closure_subset_iff, hI,
      subset_zeroLocus_iff_subset_vanishingIdeal, (gc R).u_l_u_eq_u,
      ← subset_zeroLocus_iff_subset_vanishingIdeal, ← hI]
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    I : Set R
    hI : Eq (closure t) (PrimeSpectrum.zeroLocus I)
    ⊢ And (HasSubset.Subset t (closure t)) (HasSubset.Subset t (PrimeSpectrum.zero …
  -/
  exact ⟨subset_closure, subset_zeroLocus_vanishingIdeal t⟩
  /-
    🎉 no goals
  -/


theorem vanishingIdeal_closure (t : Set (PrimeSpectrum R)) :
    vanishingIdeal (closure t) = vanishingIdeal t :=
  zeroLocus_vanishingIdeal_eq_closure t ▸ (gc R).u_l_u_eq_u t


theorem closure_singleton (x) : closure ({x} : Set (PrimeSpectrum R)) = zeroLocus x.asIdeal := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Eq (closure (Singleton.singleton x)) (PrimeSpectrum.zeroLocus ↑x.asIdeal)
  -/
  rw [← zeroLocus_vanishingIdeal_eq_closure, vanishingIdeal_singleton]
  /-
    🎉 no goals
  -/


theorem isClosed_singleton_iff_isMaximal (x : PrimeSpectrum R) :
    IsClosed ({x} : Set (PrimeSpectrum R)) ↔ x.asIdeal.IsMaximal := by
  rw [← closure_subset_iff_isClosed, ← zeroLocus_vanishingIdeal_eq_closure,
      vanishingIdeal_singleton]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (HasSubset.Subset (PrimeSpectrum.zeroLocus ↑x.asIdeal) (Singleton.single …
  -/
  constructor <;> intro H
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      x : PrimeSpectrum R
      H : HasSubset.Subset (PrimeSpectrum.zeroLocus ↑x.asIdeal) (Singleton.singleton …
      ⊢ x.asIdeal.IsMaximal
    -/
  · rcases x.asIdeal.exists_le_maximal x.2.1 with ⟨m, hm, hxm⟩
    /-
      case mp.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      x : PrimeSpectrum R
      H : HasSubset.Subset (PrimeSpectrum.zeroLocus ↑x.asIdeal) (Singleton.singleton …
      m : Ideal R
      hm : m.IsMaximal
      hxm : LE.le x.asIdeal m
      ⊢ x.asIdeal.IsMaximal
    -/
    exact (congr_arg asIdeal (@H ⟨m, hm.isPrime⟩ hxm)) ▸ hm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      x : PrimeSpectrum R
      H : x.asIdeal.IsMaximal
      ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus ↑x.asIdeal) (Singleton.singleton x)
    -/
  · exact fun p hp ↦ PrimeSpectrum.ext (H.eq_of_le p.2.1 hp).symm
    /-
      🎉 no goals
    -/


theorem isRadical_vanishingIdeal (s : Set (PrimeSpectrum R)) : (vanishingIdeal s).IsRadical := by
  rw [← vanishingIdeal_closure, ← zeroLocus_vanishingIdeal_eq_closure,
    vanishingIdeal_zeroLocus_eq_radical]
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    ⊢ (PrimeSpectrum.vanishingIdeal s).radical.IsRadical
  -/
  apply Ideal.radical_isRadical
  /-
    🎉 no goals
  -/


theorem zeroLocus_eq_iff {I J : Ideal R} :
    zeroLocus (I : Set R) = zeroLocus J ↔ I.radical = J.radical := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (Eq (PrimeSpectrum.zeroLocus ↑I) (PrimeSpectrum.zeroLocus ↑J)) (Eq I.rad …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      ⊢ Eq (PrimeSpectrum.zeroLocus ↑I) (PrimeSpectrum.zeroLocus ↑J) → Eq I.radical  …
    -/
  · intro h; simp_rw [← vanishingIdeal_zeroLocus_eq_radical, h]
             /-
               🎉 no goals
             -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I J : Ideal R
      ⊢ Eq I.radical J.radical → Eq (PrimeSpectrum.zeroLocus ↑I) (PrimeSpectrum.zero …
    -/
  · intro h; rw [← zeroLocus_radical, h, zeroLocus_radical]
             /-
               🎉 no goals
             -/


theorem vanishingIdeal_anti_mono_iff {s t : Set (PrimeSpectrum R)} (ht : IsClosed t) :
    s ⊆ t ↔ vanishingIdeal t ≤ vanishingIdeal s :=
  ⟨vanishingIdeal_anti_mono, fun h => by
    /-
      R : Type u
      inst✝ : CommSemiring R
      s t : Set (PrimeSpectrum R)
      ht : IsClosed t
      h : LE.le (PrimeSpectrum.vanishingIdeal t) (PrimeSpectrum.vanishingIdeal s)
      ⊢ HasSubset.Subset s t
    -/
    rw [← ht.closure_subset_iff, ← ht.closure_eq]
    /-
      R : Type u
      inst✝ : CommSemiring R
      s t : Set (PrimeSpectrum R)
      ht : IsClosed t
      h : LE.le (PrimeSpectrum.vanishingIdeal t) (PrimeSpectrum.vanishingIdeal s)
      ⊢ HasSubset.Subset (closure s) (closure t)
    -/
                                              /-
                                                🎉 no goals
                                              -/
    convert ← zeroLocus_anti_mono_ideal h <;> apply zeroLocus_vanishingIdeal_eq_closure⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem vanishingIdeal_strict_anti_mono_iff {s t : Set (PrimeSpectrum R)} (hs : IsClosed s)
    (ht : IsClosed t) : s ⊂ t ↔ vanishingIdeal t < vanishingIdeal s := by
  rw [Set.ssubset_def, vanishingIdeal_anti_mono_iff hs, vanishingIdeal_anti_mono_iff ht,
    lt_iff_le_not_le]


/-- The antitone order embedding of closed subsets of `Spec R` into ideals of `R`. -/
def closedsEmbedding (R : Type*) [CommSemiring R] :
    (TopologicalSpace.Closeds <| PrimeSpectrum R)ᵒᵈ ↪o Ideal R :=
  OrderEmbedding.ofMapLEIff (fun s => vanishingIdeal ↑(OrderDual.ofDual s)) fun s _ =>
    (vanishingIdeal_anti_mono_iff s.2).symm


theorem t1Space_iff_isField [IsDomain R] : T1Space (PrimeSpectrum R) ↔ IsField R := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : IsDomain R
    ⊢ Iff (T1Space (PrimeSpectrum R)) (IsField R)
  -/
  refine ⟨?_, fun h => ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsDomain R
      ⊢ T1Space (PrimeSpectrum R) → IsField R
    -/
  · intro h
    /-
      case refine_1
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsDomain R
      h : T1Space (PrimeSpectrum R)
      ⊢ IsField R
    -/
    have hbot : Ideal.IsPrime (⊥ : Ideal R) := Ideal.bot_prime
    exact
      Classical.not_not.1
        (mt
          (Ring.ne_bot_of_isMaximal_of_not_isField <|
            (isClosed_singleton_iff_isMaximal _).1 (T1Space.t1 ⟨⊥, hbot⟩))
          (by aesop))
    /-
      case refine_2
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsDomain R
      h : IsField R
      ⊢ T1Space (PrimeSpectrum R)
    -/
  · refine ⟨fun x => (isClosed_singleton_iff_isMaximal x).2 ?_⟩
    /-
      case refine_2
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsDomain R
      h : IsField R
      x : PrimeSpectrum R
      ⊢ x.asIdeal.IsMaximal
    -/
    by_cases hx : x.asIdeal = ⊥
      /-
        case pos
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : IsDomain R
        h : IsField R
        x : PrimeSpectrum R
        hx : Eq x.asIdeal Bot.bot
        ⊢ x.asIdeal.IsMaximal
      -/
    · letI := h.toSemifield
      /-
        case pos
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : IsDomain R
        h : IsField R
        x : PrimeSpectrum R
        hx : Eq x.asIdeal Bot.bot
        this : Semifield R := h.toSemifield
        ⊢ x.asIdeal.IsMaximal
      -/
      exact hx.symm ▸ Ideal.bot_isMaximal
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : IsDomain R
        h : IsField R
        x : PrimeSpectrum R
        hx : Not (Eq x.asIdeal Bot.bot)
        ⊢ x.asIdeal.IsMaximal
      -/
    · exact absurd h (Ring.not_isField_iff_exists_prime.2 ⟨x.asIdeal, ⟨hx, x.2⟩⟩)
      /-
        🎉 no goals
      -/


local notation "Z(" a ")" => zeroLocus (a : Set R)


theorem isIrreducible_zeroLocus_iff_of_radical (I : Ideal R) (hI : I.IsRadical) :
    IsIrreducible (zeroLocus (I : Set R)) ↔ I.IsPrime := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsRadical
    ⊢ Iff (IsIrreducible (PrimeSpectrum.zeroLocus ↑I)) I.IsPrime
  -/
  rw [Ideal.isPrime_iff, IsIrreducible]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsRadical
    ⊢ Iff (And (PrimeSpectrum.zeroLocus ↑I).Nonempty (IsPreirreducible (PrimeSpect …
  -/
  apply and_congr
    /-
      case h₁
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      hI : I.IsRadical
      ⊢ Iff (PrimeSpectrum.zeroLocus ↑I).Nonempty (Ne I Top.top)
    -/
  · rw [Set.nonempty_iff_ne_empty, Ne, zeroLocus_empty_iff_eq_top]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      hI : I.IsRadical
      ⊢ Iff (IsPreirreducible (PrimeSpectrum.zeroLocus ↑I)) (∀ {x y : R}, Membership …
    -/
  · trans ∀ x y : Ideal R, Z(I) ⊆ Z(x) ∪ Z(y) → Z(I) ⊆ Z(x) ∨ Z(I) ⊆ Z(y)
      /-
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        hI : I.IsRadical
        ⊢ Iff (IsPreirreducible (PrimeSpectrum.zeroLocus ↑I)) (∀ (x y : Ideal R), HasS …
      -/
    · simp_rw [isPreirreducible_iff_isClosed_union_isClosed, isClosed_iff_zeroLocus_ideal]
      /-
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        hI : I.IsRadical
        ⊢ Iff (∀ (z₁ z₂ : Set (PrimeSpectrum R)), (Exists fun I => Eq z₁ (PrimeSpectru …
      -/
      constructor
        /-
          case mp
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          ⊢ (∀ (z₁ z₂ : Set (PrimeSpectrum R)), (Exists fun I => Eq z₁ (PrimeSpectrum.ze …
        -/
      · rintro h x y
        /-
          case mp
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          h : ∀ (z₁ z₂ : Set (PrimeSpectrum R)), (Exists fun I => Eq z₁ (PrimeSpectrum.z …
          x y : Ideal R
          ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (Union.union (PrimeSpectrum.ze …
        -/
        exact h _ _ ⟨x, rfl⟩ ⟨y, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          ⊢ (∀ (x y : Ideal R), HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (Union.uni …
        -/
      · rintro h _ _ ⟨x, rfl⟩ ⟨y, rfl⟩
        /-
          case mpr.intro.intro
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          h : ∀ (x y : Ideal R), HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (Union.un …
          x y : Ideal R
          ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (Union.union (PrimeSpectrum.ze …
        -/
        exact h x y
        /-
          🎉 no goals
        -/
    · simp_rw [← zeroLocus_inf, subset_zeroLocus_iff_le_vanishingIdeal,
        vanishingIdeal_zeroLocus_eq_radical, hI.radical]
      /-
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        hI : I.IsRadical
        ⊢ Iff (∀ (x y : Ideal R), LE.le (Min.min x y) I → Or (LE.le x I) (LE.le y I))  …
      -/
      constructor
      · simp_rw [← SetLike.mem_coe, ← Set.singleton_subset_iff, ← Ideal.span_le, ←
          Ideal.span_singleton_mul_span_singleton]
        /-
          case mp
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          ⊢ (∀ (x y : Ideal R), LE.le (Min.min x y) I → Or (LE.le x I) (LE.le y I)) → ∀  …
        -/
        refine fun h x y h' => h _ _ ?_
        /-
          case mp
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          h : ∀ (x y : Ideal R), LE.le (Min.min x y) I → Or (LE.le x I) (LE.le y I)
          x y : R
          h' : LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) (Ideal.span (Single …
          ⊢ LE.le (Min.min (Ideal.span (Singleton.singleton x)) (Ideal.span (Singleton.s …
        -/
        rw [← hI.radical_le_iff] at h' ⊢
        /-
          case mp
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          h : ∀ (x y : Ideal R), LE.le (Min.min x y) I → Or (LE.le x I) (LE.le y I)
          x y : R
          h' : LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) (Ideal.span (Single …
          ⊢ LE.le (Min.min (Ideal.span (Singleton.singleton x)) (Ideal.span (Singleton.s …
        -/
        simpa only [Ideal.radical_inf, Ideal.radical_mul] using h'
        /-
          🎉 no goals
        -/
        /-
          case mpr
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          ⊢ (∀ {x y : R}, Membership.mem I (HMul.hMul x y) → Or (Membership.mem I x) (Me …
        -/
      · simp_rw [or_iff_not_imp_left, SetLike.not_le_iff_exists]
        /-
          case mpr
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          ⊢ (∀ {x y : R}, Membership.mem I (HMul.hMul x y) → Not (Membership.mem I x) →  …
        -/
        rintro h s t h' ⟨x, hx, hx'⟩ y hy
        /-
          case mpr.intro.intro
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          hI : I.IsRadical
          h : ∀ {x y : R}, Membership.mem I (HMul.hMul x y) → Not (Membership.mem I x) → …
          s t : Ideal R
          h' : LE.le (Min.min s t) I
          x : R
          hx : Membership.mem s x
          hx' : Not (Membership.mem I x)
          y : R
          hy : Membership.mem t y
          ⊢ Membership.mem I y
        -/
        exact h (h' ⟨Ideal.mul_mem_right _ _ hx, Ideal.mul_mem_left _ _ hy⟩) hx'
        /-
          🎉 no goals
        -/


theorem isIrreducible_zeroLocus_iff (I : Ideal R) :
    IsIrreducible (zeroLocus (I : Set R)) ↔ I.radical.IsPrime :=
  zeroLocus_radical I ▸ isIrreducible_zeroLocus_iff_of_radical _ I.radical_isRadical


theorem isIrreducible_iff_vanishingIdeal_isPrime {s : Set (PrimeSpectrum R)} :
    IsIrreducible s ↔ (vanishingIdeal s).IsPrime := by
  rw [← isIrreducible_iff_closure, ← zeroLocus_vanishingIdeal_eq_closure,
    isIrreducible_zeroLocus_iff_of_radical _ (isRadical_vanishingIdeal s)]


lemma vanishingIdeal_isIrreducible :
    vanishingIdeal (R := R) '' {s | IsIrreducible s} = {P | P.IsPrime} :=
  Set.ext fun I ↦ ⟨fun ⟨_, hs, e⟩ ↦ e ▸ isIrreducible_iff_vanishingIdeal_isPrime.mp hs,
    fun h ↦ ⟨zeroLocus I, (isIrreducible_zeroLocus_iff_of_radical _ h.isRadical).mpr h,
      (vanishingIdeal_zeroLocus_eq_radical I).trans h.radical⟩⟩


lemma vanishingIdeal_isClosed_isIrreducible :
    vanishingIdeal (R := R) '' {s | IsClosed s ∧ IsIrreducible s} = {P | P.IsPrime} := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (Set.image PrimeSpectrum.vanishingIdeal (setOf fun s => And (IsClosed s)  …
  -/
  refine (subset_antisymm ?_ ?_).trans vanishingIdeal_isIrreducible
    /-
      case refine_1
      R : Type u
      inst✝ : CommSemiring R
      ⊢ HasSubset.Subset (Set.image PrimeSpectrum.vanishingIdeal (setOf fun s => And …
    -/
  · exact Set.image_subset _ fun _ ↦ And.right
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u
    inst✝ : CommSemiring R
    ⊢ HasSubset.Subset (Set.image PrimeSpectrum.vanishingIdeal (setOf fun s => IsI …
  -/
  rintro _ ⟨s, hs, rfl⟩
  /-
    case refine_2.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : Membership.mem (setOf fun s => IsIrreducible s) s
    ⊢ Membership.mem (Set.image PrimeSpectrum.vanishingIdeal (setOf fun s => And ( …
  -/
  exact ⟨closure s, ⟨isClosed_closure, hs.closure⟩, vanishingIdeal_closure s⟩
  /-
    🎉 no goals
  -/


instance irreducibleSpace [IsDomain R] : IrreducibleSpace (PrimeSpectrum R) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : IsDomain R
    ⊢ IrreducibleSpace (PrimeSpectrum R)
  -/
  rw [irreducibleSpace_def, Set.top_eq_univ, ← zeroLocus_bot, isIrreducible_zeroLocus_iff]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : IsDomain R
    ⊢ Bot.bot.radical.IsPrime
  -/
  simpa using Ideal.bot_prime
  /-
    🎉 no goals
  -/


instance quasiSober : QuasiSober (PrimeSpectrum R) :=
  ⟨fun {S} h₁ h₂ =>
    ⟨⟨_, isIrreducible_iff_vanishingIdeal_isPrime.1 h₁⟩, by
      /-
        R : Type u
        S✝ : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S✝
        S : Set (PrimeSpectrum R)
        h₁ : IsIrreducible S
        h₂ : IsClosed S
        ⊢ IsGenericPoint { asIdeal := PrimeSpectrum.vanishingIdeal S, isPrime := ⋯ } S
      -/
      rw [IsGenericPoint, closure_singleton, zeroLocus_vanishingIdeal_eq_closure, h₂.closure_eq]⟩⟩
      /-
        🎉 no goals
      -/


/-- The prime spectrum of a commutative (semi)ring is a compact topological space. -/
instance compactSpace : CompactSpace (PrimeSpectrum R) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    ⊢ CompactSpace (PrimeSpectrum R)
  -/
  refine compactSpace_of_finite_subfamily_closed fun S S_closed S_empty ↦ ?_
  /-
    R : Type u
    S✝ : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S✝
    ι✝ : Type u
    S : ι✝ → Set (PrimeSpectrum R)
    S_closed : ∀ (i : ι✝), IsClosed (S i)
    S_empty : Eq (Set.iInter fun i => S i) EmptyCollection.emptyCollection
    ⊢ Exists fun u => Eq (Set.iInter fun i => Set.iInter fun h => S i) EmptyCollec …
  -/
  choose I hI using fun i ↦ (isClosed_iff_zeroLocus_ideal (S i)).mp (S_closed i)
  /-
    R : Type u
    S✝ : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S✝
    ι✝ : Type u
    S : ι✝ → Set (PrimeSpectrum R)
    S_closed : ∀ (i : ι✝), IsClosed (S i)
    S_empty : Eq (Set.iInter fun i => S i) EmptyCollection.emptyCollection
    I : ι✝ → Ideal R
    hI : ∀ (i : ι✝), Eq (S i) (PrimeSpectrum.zeroLocus ↑(I i))
    ⊢ Exists fun u => Eq (Set.iInter fun i => Set.iInter fun h => S i) EmptyCollec …
  -/
  simp_rw [hI, ← zeroLocus_iSup, zeroLocus_empty_iff_eq_top, ← top_le_iff] at S_empty ⊢
  /-
    R : Type u
    S✝ : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S✝
    ι✝ : Type u
    S : ι✝ → Set (PrimeSpectrum R)
    S_closed : ∀ (i : ι✝), IsClosed (S i)
    I : ι✝ → Ideal R
    hI : ∀ (i : ι✝), Eq (S i) (PrimeSpectrum.zeroLocus ↑(I i))
    S_empty : LE.le Top.top (iSup fun i => I i)
    ⊢ Exists fun u => LE.le Top.top (iSup fun i => iSup fun i_1 => I i)
  -/
  exact Ideal.isCompactElement_top.exists_finset_of_le_iSup _ _ S_empty
  /-
    🎉 no goals
  -/


/-- The prime spectrum of a semiring has discrete Zariski topology iff it is finite and
all primes are maximal. -/
theorem discreteTopology_iff_finite_and_isPrime_imp_isMaximal : DiscreteTopology (PrimeSpectrum R) ↔
    Finite (PrimeSpectrum R) ∧ ∀ I : Ideal R, I.IsPrime → I.IsMaximal :=
  ⟨fun _ ↦ ⟨finite_of_compact_of_discrete, fun I hI ↦ (isClosed_singleton_iff_isMaximal ⟨I, hI⟩).mp
    <| discreteTopology_iff_forall_isClosed.mp ‹_› _⟩, fun ⟨_, h⟩ ↦ .of_finite_of_isClosed_singleton
    fun p ↦ (isClosed_singleton_iff_isMaximal p).mpr <| h _ p.2⟩


/-- The prime spectrum of a semiring has discrete Zariski topology iff there are only
finitely many maximal ideals and their intersection is contained in the nilradical. -/
theorem discreteTopology_iff_finite_isMaximal_and_sInf_le_nilradical :
    letI s := {I : Ideal R | I.IsMaximal}
    DiscreteTopology (PrimeSpectrum R) ↔ Finite s ∧ sInf s ≤ nilradical R :=
  discreteTopology_iff_finite_and_isPrime_imp_isMaximal.trans <| by
    /-
      R : Type u
      inst✝ : CommSemiring R
      ⊢ Iff (And (Finite (PrimeSpectrum R)) (∀ (I : Ideal R), I.IsPrime → I.IsMaxima …
    -/
    rw [(equivSubtype R).finite_iff, ← Set.coe_setOf, Set.finite_coe_iff, Set.finite_coe_iff]
    refine ⟨fun h ↦ ⟨h.1.subset fun _ h ↦ h.isPrime, nilradical_eq_sInf R ▸ sInf_le_sInf h.2⟩,
      fun ⟨fin, le⟩ ↦ ?_⟩
    have hpm (I : Ideal R) (hI : I.IsPrime): I.IsMaximal := by
      replace le := le.trans (nilradical_le_prime I)
      rw [← fin.coe_toFinset, ← Finset.inf_id_eq_sInf, hI.inf_le'] at le
      have ⟨M, hM, hMI⟩ := le
      rw [fin.mem_toFinset] at hM
      rwa [← hM.eq_of_le hI.1 hMI]
    /-
      R : Type u
      inst✝ : CommSemiring R
      x✝ : And (setOf fun I => I.IsMaximal).Finite (LE.le (InfSet.sInf (setOf fun I  …
      fin : (setOf fun I => I.IsMaximal).Finite
      le : LE.le (InfSet.sInf (setOf fun I => I.IsMaximal)) (nilradical R)
      hpm : ∀ (I : Ideal R), I.IsPrime → I.IsMaximal
      ⊢ And (setOf fun x => x.IsPrime).Finite (∀ (I : Ideal R), I.IsPrime → I.IsMaxi …
    -/
    exact ⟨fin.subset hpm, hpm⟩
    /-
      🎉 no goals
    -/


theorem discreteTopology_of_toLocalization_surjective
    (surj : Function.Surjective (toPiLocalization R)) :
    DiscreteTopology (PrimeSpectrum R) :=
  discreteTopology_iff_finite_and_isPrime_imp_isMaximal.mpr ⟨finite_of_toPiLocalization_surjective
    surj, fun I prime ↦ isMaximal_of_toPiLocalization_surjective surj ⟨I, prime⟩⟩


/-- The continuous function between prime spectra of commutative (semi)rings induced by a ring
homomorphism. -/
def comap (f : R →+* S) : C(PrimeSpectrum S, PrimeSpectrum R) where
  toFun := f.specComap
  continuous_toFun := by
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      S' : Type u_1
      inst✝ : CommSemiring S'
      f : RingHom R S
      ⊢ Continuous f.specComap
    -/
    simp only [continuous_iff_isClosed, isClosed_iff_zeroLocus]
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      S' : Type u_1
      inst✝ : CommSemiring S'
      f : RingHom R S
      ⊢ ∀ (s : Set (PrimeSpectrum R)), (Exists fun s_1 => Eq s (PrimeSpectrum.zeroLo …
    -/
    rintro _ ⟨s, rfl⟩
    /-
      case intro
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      S' : Type u_1
      inst✝ : CommSemiring S'
      f : RingHom R S
      s : Set R
      ⊢ Exists fun s_1 => Eq (Set.preimage f.specComap (PrimeSpectrum.zeroLocus s))  …
    -/
    exact ⟨_, preimage_specComap_zeroLocus_aux f s⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem comap_asIdeal (y : PrimeSpectrum S) : (comap f y).asIdeal = Ideal.comap f y.asIdeal :=
  rfl


@[simp]
theorem comap_id : comap (RingHom.id R) = ContinuousMap.id _ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (PrimeSpectrum.comap (RingHom.id R)) (ContinuousMap.id (PrimeSpectrum R))
  -/
  ext
  /-
    case h.asIdeal.h
    R : Type u
    inst✝ : CommSemiring R
    a✝ : PrimeSpectrum R
    x✝ : R
    ⊢ Iff (Membership.mem ((PrimeSpectrum.comap (RingHom.id R)) a✝).asIdeal x✝) (M …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_comp (f : R →+* S) (g : S →+* S') : comap (g.comp f) = (comap f).comp (comap g) :=
  rfl


theorem comap_comp_apply (f : R →+* S) (g : S →+* S') (x : PrimeSpectrum S') :
    PrimeSpectrum.comap (g.comp f) x = (PrimeSpectrum.comap f) (PrimeSpectrum.comap g x) :=
  rfl


@[simp]
theorem preimage_comap_zeroLocus (s : Set R) : comap f ⁻¹' zeroLocus s = zeroLocus (f '' s) :=
  preimage_specComap_zeroLocus_aux f s


theorem comap_injective_of_surjective (f : R →+* S) (hf : Function.Surjective f) :
    Function.Injective (comap f) := fun _ _ h => specComap_injective_of_surjective _ hf h


theorem localization_comap_isInducing [Algebra R S] (M : Submonoid R) [IsLocalization M S] :
    IsInducing (comap (algebraMap R S)) := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    ⊢ Topology.IsInducing ⇑(PrimeSpectrum.comap (algebraMap R S))
  -/
  refine ⟨TopologicalSpace.ext_isClosed fun Z ↦ ?_⟩
  simp_rw [isClosed_induced_iff, isClosed_iff_zeroLocus, @eq_comm _ _ (zeroLocus _),
    exists_exists_eq_and, preimage_comap_zeroLocus]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    Z : Set (PrimeSpectrum S)
    ⊢ Iff (Exists fun s => Eq (PrimeSpectrum.zeroLocus s) Z) (Exists fun a => Eq ( …
  -/
  constructor
    /-
      case mp
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      Z : Set (PrimeSpectrum S)
      ⊢ (Exists fun s => Eq (PrimeSpectrum.zeroLocus s) Z) → Exists fun a => Eq (Pri …
    -/
  · rintro ⟨s, rfl⟩
    /-
      case mp.intro
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      s : Set S
      ⊢ Exists fun a => Eq (PrimeSpectrum.zeroLocus (Set.image (⇑(algebraMap R S)) a …
    -/
    refine ⟨(Ideal.span s).comap (algebraMap R S), ?_⟩
    /-
      case mp.intro
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      s : Set S
      ⊢ Eq (PrimeSpectrum.zeroLocus (Set.image ⇑(algebraMap R S) ↑(Ideal.comap (alge …
    -/
    rw [← zeroLocus_span, ← zeroLocus_span s, ← Ideal.map, IsLocalization.map_comap M S]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      Z : Set (PrimeSpectrum S)
      ⊢ (Exists fun a => Eq (PrimeSpectrum.zeroLocus (Set.image (⇑(algebraMap R S))  …
    -/
  · rintro ⟨s, rfl⟩
    /-
      case mpr.intro
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      s : Set R
      ⊢ Exists fun s_1 => Eq (PrimeSpectrum.zeroLocus s_1) (PrimeSpectrum.zeroLocus  …
    -/
    exact ⟨_, rfl⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")]
alias localization_comap_inducing := localization_comap_isInducing


theorem localization_comap_injective [Algebra R S] (M : Submonoid R) [IsLocalization M S] :
    Function.Injective (comap (algebraMap R S)) :=
  fun _ _ h => localization_specComap_injective S M h


theorem localization_comap_isEmbedding [Algebra R S] (M : Submonoid R) [IsLocalization M S] :
    IsEmbedding (comap (algebraMap R S)) :=
  ⟨localization_comap_isInducing S M, localization_comap_injective S M⟩


@[deprecated (since := "2024-10-26")]
alias localization_comap_embedding := localization_comap_isEmbedding


theorem localization_comap_range [Algebra R S] (M : Submonoid R) [IsLocalization M S] :
    Set.range (comap (algebraMap R S)) = { p | Disjoint (M : Set R) p.asIdeal } :=
  localization_specComap_range ..


theorem comap_isInducing_of_surjective (hf : Surjective f) : IsInducing (comap f) where
  eq_induced := by
    simp only [TopologicalSpace.ext_iff, ← isClosed_compl_iff, isClosed_iff_zeroLocus,
      isClosed_induced_iff]
    refine fun s =>
      ⟨fun ⟨F, hF⟩ =>
        ⟨zeroLocus (f ⁻¹' F), ⟨f ⁻¹' F, rfl⟩, by
          rw [preimage_comap_zeroLocus, Function.Surjective.image_preimage hf, hF]⟩,
        ?_⟩
    /-
      R : Type u
      S : Type v
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      s : Set (PrimeSpectrum S)
      ⊢ (Exists fun t => And (Exists fun s => Eq t (PrimeSpectrum.zeroLocus s)) (Eq  …
    -/
    rintro ⟨-, ⟨F, rfl⟩, hF⟩
    /-
      case intro.intro.intro
      R : Type u
      S : Type v
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      s : Set (PrimeSpectrum S)
      F : Set R
      hF : Eq (Set.preimage (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum.zeroLocus F))  …
      ⊢ Exists fun s_1 => Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus s_1)
    -/
    exact ⟨f '' F, hF.symm.trans (preimage_comap_zeroLocus f F)⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")]
alias comap_inducing_of_surjective := comap_isInducing_of_surjective



theorem comap_singleton_isClosed_of_surjective (f : R →+* S) (hf : Function.Surjective f)
    (x : PrimeSpectrum S) (hx : IsClosed ({x} : Set (PrimeSpectrum S))) :
    IsClosed ({comap f x} : Set (PrimeSpectrum R)) :=
  haveI : x.asIdeal.IsMaximal := (isClosed_singleton_iff_isMaximal x).1 hx
  (isClosed_singleton_iff_isMaximal _).2 (Ideal.comap_isMaximal_of_surjective f hf)


theorem image_comap_zeroLocus_eq_zeroLocus_comap (hf : Surjective f) (I : Ideal S) :
    comap f '' zeroLocus I = zeroLocus (I.comap f) :=
  image_specComap_zeroLocus_eq_zeroLocus_comap _ f hf I


theorem range_comap_of_surjective (hf : Surjective f) :
    Set.range (comap f) = zeroLocus (ker f) :=
  range_specComap_of_surjective _ f hf


theorem isClosed_range_comap_of_surjective (hf : Surjective f) :
    IsClosed (Set.range (comap f)) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ IsClosed (Set.range ⇑(PrimeSpectrum.comap f))
  -/
  rw [range_comap_of_surjective _ f hf]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ IsClosed (PrimeSpectrum.zeroLocus ↑(RingHom.ker f))
  -/
  exact isClosed_zeroLocus _
  /-
    🎉 no goals
  -/


lemma isClosedEmbedding_comap_of_surjective (hf : Surjective f) : IsClosedEmbedding (comap f) where
  toIsInducing := comap_isInducing_of_surjective S f hf
  injective := comap_injective_of_surjective f hf
  isClosed_range := isClosed_range_comap_of_surjective S f hf


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_comap_of_surjective := isClosedEmbedding_comap_of_surjective


lemma primeSpectrumProd_symm_inl (x) :
    (primeSpectrumProd R S).symm (.inl x) = comap (RingHom.fst R S) x := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    x : PrimeSpectrum R
    ⊢ Eq ((PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inl x)) ((PrimeSpectrum. …
  -/
  ext; simp [Ideal.prod]
       /-
         🎉 no goals
       -/


lemma primeSpectrumProd_symm_inr (x) :
    (primeSpectrumProd R S).symm (.inr x) = comap (RingHom.snd R S) x := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    x : PrimeSpectrum S
    ⊢ Eq ((PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inr x)) ((PrimeSpectrum. …
  -/
  ext; simp [Ideal.prod]
       /-
         🎉 no goals
       -/


/-- The prime spectrum of `R × S` is homeomorphic
to the disjoint union of `PrimeSpectrum R` and `PrimeSpectrum S`. -/
noncomputable
def primeSpectrumProdHomeo :
    PrimeSpectrum (R × S) ≃ₜ PrimeSpectrum R ⊕ PrimeSpectrum S := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    ⊢ Homeomorph (PrimeSpectrum (Prod R S)) (Sum (PrimeSpectrum R) (PrimeSpectrum  …
  -/
  refine ((primeSpectrumProd R S).symm.toHomeomorphOfIsInducing ?_).symm
  refine (IsClosedEmbedding.of_continuous_injective_isClosedMap ?_
    (Equiv.injective _) ?_).isInducing
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ Continuous ⇑(PrimeSpectrum.primeSpectrumProd R S).symm
    -/
  · rw [continuous_sum_dom]
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ And (Continuous (Function.comp (⇑(PrimeSpectrum.primeSpectrumProd R S).symm) …
    -/
    simp only [Function.comp_def, primeSpectrumProd_symm_inl, primeSpectrumProd_symm_inr]
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ And (Continuous fun x => (PrimeSpectrum.comap (RingHom.fst R S)) x) (Continu …
    -/
    exact ⟨(comap _).2, (comap _).2⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ IsClosedMap ⇑(PrimeSpectrum.primeSpectrumProd R S).symm
    -/
  · rw [isClosedMap_sum]
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      ⊢ And (IsClosedMap fun a => (PrimeSpectrum.primeSpectrumProd R S).symm (Sum.in …
    -/
    constructor
      /-
        case refine_2.left
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ IsClosedMap fun a => (PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inl a)
      -/
    · simp_rw [primeSpectrumProd_symm_inl]
      /-
        case refine_2.left
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ IsClosedMap fun a => (PrimeSpectrum.comap (RingHom.fst R S)) a
      -/
      refine (isClosedEmbedding_comap_of_surjective _ _ ?_).isClosedMap
      /-
        case refine_2.left
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ Function.Surjective ⇑(RingHom.fst R S)
      -/
      exact Prod.fst_surjective
      /-
        🎉 no goals
      -/
      /-
        case refine_2.right
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ IsClosedMap fun b => (PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inr b)
      -/
    · simp_rw [primeSpectrumProd_symm_inr]
      /-
        case refine_2.right
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ IsClosedMap fun b => (PrimeSpectrum.comap (RingHom.snd R S)) b
      -/
      refine (isClosedEmbedding_comap_of_surjective _ _ ?_).isClosedMap
      /-
        case refine_2.right
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        ⊢ Function.Surjective ⇑(RingHom.snd R S)
      -/
      exact Prod.snd_surjective
      /-
        🎉 no goals
      -/


/-- `basicOpen r` is the open subset containing all prime ideals not containing `r`. -/
def basicOpen (r : R) : TopologicalSpace.Opens (PrimeSpectrum R) where
  carrier := { x | r ∉ x.asIdeal }
  is_open' := ⟨{r}, Set.ext fun _ => Set.singleton_subset_iff.trans <| Classical.not_not.symm⟩


@[simp]
theorem mem_basicOpen (f : R) (x : PrimeSpectrum R) : x ∈ basicOpen f ↔ f ∉ x.asIdeal :=
  Iff.rfl


theorem isOpen_basicOpen {a : R} : IsOpen (basicOpen a : Set (PrimeSpectrum R)) :=
  (basicOpen a).isOpen


@[simp]
theorem basicOpen_eq_zeroLocus_compl (r : R) :
    (basicOpen r : Set (PrimeSpectrum R)) = (zeroLocus {r})ᶜ :=
  Set.ext fun x => by simp only [SetLike.mem_coe, mem_basicOpen, Set.mem_compl_iff, mem_zeroLocus,
    Set.singleton_subset_iff]


@[simp]
theorem basicOpen_one : basicOpen (1 : R) = ⊤ :=
                                   /-
                                     R : Type u
                                     inst✝ : CommSemiring R
                                     ⊢ Eq ↑(PrimeSpectrum.basicOpen 1) ↑Top.top
                                   -/
  TopologicalSpace.Opens.ext <| by simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem basicOpen_zero : basicOpen (0 : R) = ⊥ :=
                                   /-
                                     R : Type u
                                     inst✝ : CommSemiring R
                                     ⊢ Eq ↑(PrimeSpectrum.basicOpen 0) ↑Bot.bot
                                   -/
  TopologicalSpace.Opens.ext <| by simp
                                   /-
                                     🎉 no goals
                                   -/


theorem basicOpen_le_basicOpen_iff (f g : R) :
    basicOpen f ≤ basicOpen g ↔ f ∈ (Ideal.span ({g} : Set R)).radical := by
  rw [← SetLike.coe_subset_coe, basicOpen_eq_zeroLocus_compl, basicOpen_eq_zeroLocus_compl,
    Set.compl_subset_compl, zeroLocus_subset_zeroLocus_singleton_iff]


theorem basicOpen_mul (f g : R) : basicOpen (f * g) = basicOpen f ⊓ basicOpen g :=
                                   /-
                                     R : Type u
                                     inst✝ : CommSemiring R
                                     f g : R
                                     ⊢ Eq ↑(PrimeSpectrum.basicOpen (HMul.hMul f g)) ↑(Min.min (PrimeSpectrum.basic …
                                   -/
  TopologicalSpace.Opens.ext <| by simp [zeroLocus_singleton_mul]
                                   /-
                                     🎉 no goals
                                   -/


theorem basicOpen_mul_le_left (f g : R) : basicOpen (f * g) ≤ basicOpen f := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : R
    ⊢ LE.le (PrimeSpectrum.basicOpen (HMul.hMul f g)) (PrimeSpectrum.basicOpen f)
  -/
  rw [basicOpen_mul f g]
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : R
    ⊢ LE.le (Min.min (PrimeSpectrum.basicOpen f) (PrimeSpectrum.basicOpen g)) (Pri …
  -/
  exact inf_le_left
  /-
    🎉 no goals
  -/


theorem basicOpen_mul_le_right (f g : R) : basicOpen (f * g) ≤ basicOpen g := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : R
    ⊢ LE.le (PrimeSpectrum.basicOpen (HMul.hMul f g)) (PrimeSpectrum.basicOpen g)
  -/
  rw [basicOpen_mul f g]
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : R
    ⊢ LE.le (Min.min (PrimeSpectrum.basicOpen f) (PrimeSpectrum.basicOpen g)) (Pri …
  -/
  exact inf_le_right
  /-
    🎉 no goals
  -/


@[simp]
theorem basicOpen_pow (f : R) (n : ℕ) (hn : 0 < n) : basicOpen (f ^ n) = basicOpen f :=
                                   /-
                                     R : Type u
                                     inst✝ : CommSemiring R
                                     f : R
                                     n : Nat
                                     hn : LT.lt 0 n
                                     ⊢ Eq ↑(PrimeSpectrum.basicOpen (HPow.hPow f n)) ↑(PrimeSpectrum.basicOpen f)
                                   -/
  TopologicalSpace.Opens.ext <| by simpa using zeroLocus_singleton_pow f n hn
                                   /-
                                     🎉 no goals
                                   -/


theorem isTopologicalBasis_basic_opens :
    TopologicalSpace.IsTopologicalBasis
      (Set.range fun r : R => (basicOpen r : Set (PrimeSpectrum R))) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.range fun r => ↑(PrimeSpectrum.basi …
  -/
  apply TopologicalSpace.isTopologicalBasis_of_isOpen_of_nhds
    /-
      case h_open
      R : Type u
      inst✝ : CommSemiring R
      ⊢ ∀ (u : Set (PrimeSpectrum R)), Membership.mem (Set.range fun r => ↑(PrimeSpe …
    -/
  · rintro _ ⟨r, rfl⟩
    /-
      case h_open.intro
      R : Type u
      inst✝ : CommSemiring R
      r : R
      ⊢ IsOpen ((fun r => ↑(PrimeSpectrum.basicOpen r)) r)
    -/
    exact isOpen_basicOpen
    /-
      🎉 no goals
    -/
    /-
      case h_nhds
      R : Type u
      inst✝ : CommSemiring R
      ⊢ ∀ (a : PrimeSpectrum R) (u : Set (PrimeSpectrum R)), Membership.mem u a → Is …
    -/
  · rintro p U hp ⟨s, hs⟩
    /-
      case h_nhds.intro
      R : Type u
      inst✝ : CommSemiring R
      p : PrimeSpectrum R
      U : Set (PrimeSpectrum R)
      hp : Membership.mem U p
      s : Set R
      hs : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
      ⊢ Exists fun v => And (Membership.mem (Set.range fun r => ↑(PrimeSpectrum.basi …
    -/
    rw [← compl_compl U, Set.mem_compl_iff, ← hs, mem_zeroLocus, Set.not_subset] at hp
    /-
      case h_nhds.intro
      R : Type u
      inst✝ : CommSemiring R
      p : PrimeSpectrum R
      U : Set (PrimeSpectrum R)
      s : Set R
      hp : Exists fun a => And (Membership.mem s a) (Not (Membership.mem (↑p.asIdeal …
      hs : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
      ⊢ Exists fun v => And (Membership.mem (Set.range fun r => ↑(PrimeSpectrum.basi …
    -/
    obtain ⟨f, hfs, hfp⟩ := hp
    /-
      case h_nhds.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      p : PrimeSpectrum R
      U : Set (PrimeSpectrum R)
      s : Set R
      hs : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
      f : R
      hfs : Membership.mem s f
      hfp : Not (Membership.mem (↑p.asIdeal) f)
      ⊢ Exists fun v => And (Membership.mem (Set.range fun r => ↑(PrimeSpectrum.basi …
    -/
    refine ⟨basicOpen f, ⟨f, rfl⟩, hfp, ?_⟩
    /-
      case h_nhds.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      p : PrimeSpectrum R
      U : Set (PrimeSpectrum R)
      s : Set R
      hs : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
      f : R
      hfs : Membership.mem s f
      hfp : Not (Membership.mem (↑p.asIdeal) f)
      ⊢ HasSubset.Subset (↑(PrimeSpectrum.basicOpen f)) U
    -/
    rw [← Set.compl_subset_compl, ← hs, basicOpen_eq_zeroLocus_compl, compl_compl]
    /-
      case h_nhds.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      p : PrimeSpectrum R
      U : Set (PrimeSpectrum R)
      s : Set R
      hs : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
      f : R
      hfs : Membership.mem s f
      hfp : Not (Membership.mem (↑p.asIdeal) f)
      ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus s) (PrimeSpectrum.zeroLocus (Singl …
    -/
    exact zeroLocus_anti_mono (Set.singleton_subset_iff.mpr hfs)
    /-
      🎉 no goals
    -/


theorem isBasis_basic_opens : TopologicalSpace.Opens.IsBasis (Set.range (@basicOpen R _)) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ TopologicalSpace.Opens.IsBasis (Set.range PrimeSpectrum.basicOpen)
  -/
  unfold TopologicalSpace.Opens.IsBasis
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.image SetLike.coe (Set.range PrimeS …
  -/
  convert isTopologicalBasis_basic_opens (R := R)
  /-
    case h.e'_3
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (Set.image SetLike.coe (Set.range PrimeSpectrum.basicOpen)) (Set.range fu …
  -/
  rw [← Set.range_comp]
  /-
    case h.e'_3
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (Set.range (Function.comp SetLike.coe PrimeSpectrum.basicOpen)) (Set.rang …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem basicOpen_eq_bot_iff (f : R) : basicOpen f = ⊥ ↔ IsNilpotent f := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : R
    ⊢ Iff (Eq (PrimeSpectrum.basicOpen f) Bot.bot) (IsNilpotent f)
  -/
  rw [← TopologicalSpace.Opens.coe_inj, basicOpen_eq_zeroLocus_compl]
  simp only [Set.eq_univ_iff_forall, Set.singleton_subset_iff, TopologicalSpace.Opens.coe_bot,
    nilpotent_iff_mem_prime, Set.compl_empty_iff, mem_zeroLocus, SetLike.mem_coe]
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : R
    ⊢ Iff (∀ (x : PrimeSpectrum R), Membership.mem x.asIdeal f) (∀ (J : Ideal R),  …
  -/
  exact ⟨fun h I hI => h ⟨I, hI⟩, fun h ⟨I, hI⟩ => h I hI⟩
  /-
    🎉 no goals
  -/


theorem localization_away_comap_range (S : Type v) [CommSemiring S] [Algebra R S] (r : R)
    [IsLocalization.Away r S] : Set.range (comap (algebraMap R S)) = basicOpen r := by
  /-
    R : Type u
    inst✝³ : CommSemiring R
    S : Type v
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (Set.range ⇑(PrimeSpectrum.comap (algebraMap R S))) ↑(PrimeSpectrum.basic …
  -/
  rw [localization_comap_range S (Submonoid.powers r)]
  /-
    R : Type u
    inst✝³ : CommSemiring R
    S : Type v
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (setOf fun p => Disjoint ↑(Submonoid.powers r) ↑p.asIdeal) ↑(PrimeSpectru …
  -/
  ext x
  simp only [mem_zeroLocus, basicOpen_eq_zeroLocus_compl, SetLike.mem_coe, Set.mem_setOf_eq,
    Set.singleton_subset_iff, Set.mem_compl_iff, disjoint_iff_inf_le]
  /-
    case h
    R : Type u
    inst✝³ : CommSemiring R
    S : Type v
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    x : PrimeSpectrum R
    ⊢ Iff (LE.le (Min.min ↑(Submonoid.powers r) ↑x.asIdeal) Bot.bot) (Not (Members …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      inst✝³ : CommSemiring R
      S : Type v
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : PrimeSpectrum R
      ⊢ LE.le (Min.min ↑(Submonoid.powers r) ↑x.asIdeal) Bot.bot → Not (Membership.m …
    -/
  · intro h₁ h₂
    /-
      case h.mp
      R : Type u
      inst✝³ : CommSemiring R
      S : Type v
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : PrimeSpectrum R
      h₁ : LE.le (Min.min ↑(Submonoid.powers r) ↑x.asIdeal) Bot.bot
      h₂ : Membership.mem x.asIdeal r
      ⊢ False
    -/
    exact h₁ ⟨Submonoid.mem_powers r, h₂⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      inst✝³ : CommSemiring R
      S : Type v
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : PrimeSpectrum R
      ⊢ Not (Membership.mem x.asIdeal r) → LE.le (Min.min ↑(Submonoid.powers r) ↑x.a …
    -/
  · rintro h₁ _ ⟨⟨n, rfl⟩, h₃⟩
    /-
      case h.mpr.intro.intro
      R : Type u
      inst✝³ : CommSemiring R
      S : Type v
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      x : PrimeSpectrum R
      h₁ : Not (Membership.mem x.asIdeal r)
      n : Nat
      h₃ : Membership.mem (↑x.asIdeal) ((fun x => HPow.hPow r x) n)
      ⊢ Membership.mem Bot.bot ((fun x => HPow.hPow r x) n)
    -/
    exact h₁ (x.2.mem_of_pow_mem _ h₃)
    /-
      🎉 no goals
    -/


theorem localization_away_isOpenEmbedding (S : Type v) [CommSemiring S] [Algebra R S] (r : R)
    [IsLocalization.Away r S] : IsOpenEmbedding (comap (algebraMap R S)) where
  toIsEmbedding := localization_comap_isEmbedding S (Submonoid.powers r)
  isOpen_range := by
    /-
      R : Type u
      inst✝³ : CommSemiring R
      S : Type v
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ IsOpen (Set.range ⇑(PrimeSpectrum.comap (algebraMap R S)))
    -/
    rw [localization_away_comap_range S r]
    /-
      R : Type u
      inst✝³ : CommSemiring R
      S : Type v
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ IsOpen ↑(PrimeSpectrum.basicOpen r)
    -/
    exact isOpen_basicOpen
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias localization_away_openEmbedding := localization_away_isOpenEmbedding


theorem isCompact_basicOpen (f : R) : IsCompact (basicOpen f : Set (PrimeSpectrum R)) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : R
    ⊢ IsCompact ↑(PrimeSpectrum.basicOpen f)
  -/
  rw [← localization_away_comap_range (Localization (Submonoid.powers f))]
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : R
    ⊢ IsCompact (Set.range ⇑(PrimeSpectrum.comap (algebraMap R (Localization (Subm …
  -/
  exact isCompact_range (map_continuous _)
  /-
    🎉 no goals
  -/


lemma comap_basicOpen (f : R →+* S) (x : R) :
    TopologicalSpace.Opens.comap (comap f) (basicOpen x) = basicOpen (f x) :=
  rfl


open TopologicalSpace in
lemma iSup_basicOpen_eq_top_iff {ι : Type*} {f : ι → R} :
    (⨆ i : ι, PrimeSpectrum.basicOpen (f i)) = ⊤ ↔ Ideal.span (Set.range f) = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_1
    f : ι → R
    ⊢ Iff (Eq (iSup fun i => PrimeSpectrum.basicOpen (f i)) Top.top) (Eq (Ideal.sp …
  -/
  rw [SetLike.ext'_iff, Opens.coe_iSup]
  simp only [PrimeSpectrum.basicOpen_eq_zeroLocus_compl, Opens.coe_top, ← Set.compl_iInter,
    ← PrimeSpectrum.zeroLocus_iUnion]
  /-
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_1
    f : ι → R
    ⊢ Iff (Eq (HasCompl.compl (PrimeSpectrum.zeroLocus (Set.iUnion fun i => Single …
  -/
  rw [← PrimeSpectrum.zeroLocus_empty_iff_eq_top, compl_involutive.eq_iff]
  /-
    R : Type u
    inst✝ : CommSemiring R
    ι : Type u_1
    f : ι → R
    ⊢ Iff (Eq (PrimeSpectrum.zeroLocus (Set.iUnion fun i => Singleton.singleton (f …
  -/
  simp only [Set.iUnion_singleton_eq_range,  Set.compl_univ, PrimeSpectrum.zeroLocus_span]
  /-
    🎉 no goals
  -/


lemma iSup_basicOpen_eq_top_iff' {s : Set R} :
    (⨆ i ∈ s, PrimeSpectrum.basicOpen i) = ⊤ ↔ Ideal.span s = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Iff (Eq (iSup fun i => iSup fun h => PrimeSpectrum.basicOpen i) Top.top) (Eq …
  -/
  conv_rhs => rw [← Subtype.range_val (s := s), ← iSup_basicOpen_eq_top_iff]
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Iff (Eq (iSup fun i => iSup fun h => PrimeSpectrum.basicOpen i) Top.top) (Eq …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isLocalization_away_iff_atPrime_of_basicOpen_eq_singleton [Algebra R S]
    {f : R} {p : PrimeSpectrum R} (h : (basicOpen f).1 = {p}) :
    IsLocalization.Away f S ↔ IsLocalization.AtPrime S p.1 :=
  have : IsLocalization.AtPrime (Localization.Away f) p.1 := by
    refine .of_le_of_exists_dvd (.powers f) _
      (Submonoid.powers_le.mpr <| by apply h ▸ Set.mem_singleton p) fun r hr ↦ ?_
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      f : R
      p : PrimeSpectrum R
      h : Eq (PrimeSpectrum.basicOpen f).carrier (Singleton.singleton p)
      r : R
      hr : Membership.mem p.asIdeal.primeCompl r
      ⊢ Exists fun m => And (Membership.mem (Submonoid.powers f) m) (Dvd.dvd r m)
    -/
    contrapose! hr
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      f : R
      p : PrimeSpectrum R
      h : Eq (PrimeSpectrum.basicOpen f).carrier (Singleton.singleton p)
      r : R
      hr : ∀ (m : R), Membership.mem (Submonoid.powers f) m → Not (Dvd.dvd r m)
      ⊢ Not (Membership.mem p.asIdeal.primeCompl r)
    -/
    simp_rw [← Ideal.mem_span_singleton] at hr
    have ⟨q, prime, le, disj⟩ := Ideal.exists_le_prime_disjoint (Ideal.span {r})
      (.powers f) (Set.disjoint_right.mpr hr)
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      f : R
      p : PrimeSpectrum R
      h : Eq (PrimeSpectrum.basicOpen f).carrier (Singleton.singleton p)
      r : R
      hr : ∀ (m : R), Membership.mem (Submonoid.powers f) m → Not (Membership.mem (I …
      q : Ideal R
      prime : q.IsPrime
      le : LE.le (Ideal.span (Singleton.singleton r)) q
      disj : Disjoint ↑q ↑(Submonoid.powers f)
      ⊢ Not (Membership.mem p.asIdeal.primeCompl r)
    -/
    have : ⟨q, prime⟩ ∈ (basicOpen f).1 := Set.disjoint_right.mp disj (Submonoid.mem_powers f)
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      f : R
      p : PrimeSpectrum R
      h : Eq (PrimeSpectrum.basicOpen f).carrier (Singleton.singleton p)
      r : R
      hr : ∀ (m : R), Membership.mem (Submonoid.powers f) m → Not (Membership.mem (I …
      q : Ideal R
      prime : q.IsPrime
      le : LE.le (Ideal.span (Singleton.singleton r)) q
      disj : Disjoint ↑q ↑(Submonoid.powers f)
      this : Membership.mem (PrimeSpectrum.basicOpen f).carrier { asIdeal := q, isPr …
      ⊢ Not (Membership.mem p.asIdeal.primeCompl r)
    -/
    rw [h, Set.mem_singleton_iff] at this
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      f : R
      p : PrimeSpectrum R
      h : Eq (PrimeSpectrum.basicOpen f).carrier (Singleton.singleton p)
      r : R
      hr : ∀ (m : R), Membership.mem (Submonoid.powers f) m → Not (Membership.mem (I …
      q : Ideal R
      prime : q.IsPrime
      le : LE.le (Ideal.span (Singleton.singleton r)) q
      disj : Disjoint ↑q ↑(Submonoid.powers f)
      this : Eq { asIdeal := q, isPrime := prime } p
      ⊢ Not (Membership.mem p.asIdeal.primeCompl r)
    -/
    rw [← this]
    /-
      R : Type u
      S : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      f : R
      p : PrimeSpectrum R
      h : Eq (PrimeSpectrum.basicOpen f).carrier (Singleton.singleton p)
      r : R
      hr : ∀ (m : R), Membership.mem (Submonoid.powers f) m → Not (Membership.mem (I …
      q : Ideal R
      prime : q.IsPrime
      le : LE.le (Ideal.span (Singleton.singleton r)) q
      disj : Disjoint ↑q ↑(Submonoid.powers f)
      this : Eq { asIdeal := q, isPrime := prime } p
      ⊢ Not (Membership.mem { asIdeal := q, isPrime := prime }.asIdeal.primeCompl r)
    -/
    exact not_not.mpr (q.span_singleton_le_iff_mem.mp le)
    /-
      🎉 no goals
    -/
  IsLocalization.isLocalization_iff_of_isLocalization _ _ (Localization.Away f)


theorem toPiLocalization_surjective_of_discreteTopology :
    Function.Surjective (toPiLocalization R) := fun x ↦ by
  have (p : PrimeSpectrum R) : ∃ f, (basicOpen f : Set _) = {p} :=
    have ⟨_, ⟨f, rfl⟩, hpf, hfp⟩ := isTopologicalBasis_basic_opens.isOpen_iff.mp
      (isOpen_discrete {p}) p rfl
    ⟨f, hfp.antisymm <| Set.singleton_subset_iff.mpr hpf⟩
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : DiscreteTopology (PrimeSpectrum R)
    x : PrimeSpectrum.PiLocalization R
    this : ∀ (p : PrimeSpectrum R), Exists fun f => Eq (↑(PrimeSpectrum.basicOpen  …
    ⊢ Exists fun a => Eq ((PrimeSpectrum.toPiLocalization R) a) x
  -/
  choose f hf using this
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : DiscreteTopology (PrimeSpectrum R)
    x : PrimeSpectrum.PiLocalization R
    f : PrimeSpectrum R → R
    hf : ∀ (p : PrimeSpectrum R), Eq (↑(PrimeSpectrum.basicOpen (f p))) (Singleton …
    ⊢ Exists fun a => Eq ((PrimeSpectrum.toPiLocalization R) a) x
  -/
  let e := Equiv.ofInjective f fun p q eq ↦ Set.singleton_injective (hf p ▸ eq ▸ hf q)
  have loc a : IsLocalization.AtPrime (Localization.Away a.1) (e.symm a).1 :=
    (isLocalization_away_iff_atPrime_of_basicOpen_eq_singleton <| hf _).mp <| by
      simp_rw [e, Equiv.apply_ofInjective_symm]; infer_instance
  let algE a := IsLocalization.algEquiv (e.symm a).1.primeCompl
    (Localization.AtPrime (e.symm a).1) (Localization.Away a.1)
  have span_eq : Ideal.span (Set.range f) = ⊤ := iSup_basicOpen_eq_top_iff.mp <| top_unique
    fun p _ ↦ TopologicalSpace.Opens.mem_iSup.mpr ⟨p, (hf p).ge rfl⟩
  replace hf a : (basicOpen a.1 : Set _) = {e.symm a} := by
    simp_rw [e, ← hf, Equiv.apply_ofInjective_symm]
  obtain ⟨r, eq, -⟩ := Localization.existsUnique_algebraMap_eq_of_span_eq_top _ span_eq
    (fun a ↦ algE a (x _)) fun a b ↦ by
      obtain rfl | ne := eq_or_ne a b; · rfl
      have ⟨n, hn⟩ : IsNilpotent (a * b : R) := (basicOpen_eq_bot_iff _).mp <| by
        simp_rw [basicOpen_mul, SetLike.ext'_iff, TopologicalSpace.Opens.coe_inf, hf]
        exact bot_unique (fun _ ⟨ha, hb⟩ ↦ ne <| e.symm.injective (ha.symm.trans hb))
      have := IsLocalization.subsingleton (M := .powers (a * b : R))
        (S := Localization.Away (a * b : R)) <| hn ▸ ⟨n, rfl⟩
      apply Subsingleton.elim
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : DiscreteTopology (PrimeSpectrum R)
    x : PrimeSpectrum.PiLocalization R
    f : PrimeSpectrum R → R
    hf✝ : ∀ (p : PrimeSpectrum R), Eq (↑(PrimeSpectrum.basicOpen (f p))) (Singleto …
    e : Equiv (PrimeSpectrum R) ↑(Set.range f) := Equiv.ofInjective f ⋯
    loc : ∀ (a : ↑(Set.range f)), IsLocalization.AtPrime (Localization.Away ↑a) (e …
    algE : (a : ↑(Set.range f)) → AlgEquiv R (Localization.AtPrime (e.symm a).asId …
    span_eq : Eq (Ideal.span (Set.range f)) Top.top
    hf : ∀ (a : ↑(Set.range f)), Eq (↑(PrimeSpectrum.basicOpen ↑a)) (Singleton.sin …
    r : R
    eq : ∀ (a : ↑(Set.range f)), Eq ((algebraMap R (Localization.Away ↑a)) r) ((al …
    ⊢ Exists fun a => Eq ((PrimeSpectrum.toPiLocalization R) a) x
  -/
  refine ⟨r, funext fun I ↦ ?_⟩
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : DiscreteTopology (PrimeSpectrum R)
    x : PrimeSpectrum.PiLocalization R
    f : PrimeSpectrum R → R
    hf✝ : ∀ (p : PrimeSpectrum R), Eq (↑(PrimeSpectrum.basicOpen (f p))) (Singleto …
    e : Equiv (PrimeSpectrum R) ↑(Set.range f) := Equiv.ofInjective f ⋯
    loc : ∀ (a : ↑(Set.range f)), IsLocalization.AtPrime (Localization.Away ↑a) (e …
    algE : (a : ↑(Set.range f)) → AlgEquiv R (Localization.AtPrime (e.symm a).asId …
    span_eq : Eq (Ideal.span (Set.range f)) Top.top
    hf : ∀ (a : ↑(Set.range f)), Eq (↑(PrimeSpectrum.basicOpen ↑a)) (Singleton.sin …
    r : R
    eq : ∀ (a : ↑(Set.range f)), Eq ((algebraMap R (Localization.Away ↑a)) r) ((al …
    I : PrimeSpectrum R
    ⊢ Eq ((PrimeSpectrum.toPiLocalization R) r I) (x I)
  -/
  have := eq (e I)
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : DiscreteTopology (PrimeSpectrum R)
    x : PrimeSpectrum.PiLocalization R
    f : PrimeSpectrum R → R
    hf✝ : ∀ (p : PrimeSpectrum R), Eq (↑(PrimeSpectrum.basicOpen (f p))) (Singleto …
    e : Equiv (PrimeSpectrum R) ↑(Set.range f) := Equiv.ofInjective f ⋯
    loc : ∀ (a : ↑(Set.range f)), IsLocalization.AtPrime (Localization.Away ↑a) (e …
    algE : (a : ↑(Set.range f)) → AlgEquiv R (Localization.AtPrime (e.symm a).asId …
    span_eq : Eq (Ideal.span (Set.range f)) Top.top
    hf : ∀ (a : ↑(Set.range f)), Eq (↑(PrimeSpectrum.basicOpen ↑a)) (Singleton.sin …
    r : R
    eq : ∀ (a : ↑(Set.range f)), Eq ((algebraMap R (Localization.Away ↑a)) r) ((al …
    I : PrimeSpectrum R
    this : Eq ((algebraMap R (Localization.Away ↑(e I))) r) ((algE (e I)) (x (e.sy …
    ⊢ Eq ((PrimeSpectrum.toPiLocalization R) r I) (x I)
  -/
  rwa [← AlgEquiv.symm_apply_eq, AlgEquiv.commutes, e.symm_apply_apply] at this
  /-
    🎉 no goals
  -/


theorem maximalSpectrumToPiLocalization_surjective_of_discreteTopology :
    Function.Surjective (MaximalSpectrum.toPiLocalization R) := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : DiscreteTopology (PrimeSpectrum R)
    ⊢ Function.Surjective ⇑(MaximalSpectrum.toPiLocalization R)
  -/
  rw [← piLocalizationToMaximal_comp_toPiLocalization]
  exact (piLocalizationToMaximal_surjective R).comp
    (toPiLocalization_surjective_of_discreteTopology R)


/-- If the prime spectrum of a commutative semiring R has discrete Zariski topology, then R is
canonically isomorphic to the product of its localizations at the (finitely many) maximal ideals. -/
@[stacks 00JA
"See also `PrimeSpectrum.discreteTopology_iff_finite_isMaximal_and_sInf_le_nilradical`."]
def MaximalSpectrum.toPiLocalizationEquivtoLocalizationEquiv :
    R ≃+* MaximalSpectrum.PiLocalization R :=
  .ofBijective _ ⟨MaximalSpectrum.toPiLocalization_injective R,
    maximalSpectrumToPiLocalization_surjective_of_discreteTopology R⟩


theorem discreteTopology_iff_toPiLocalization_surjective {R} [CommSemiring R] :
    DiscreteTopology (PrimeSpectrum R) ↔ Function.Surjective (toPiLocalization R) :=
  ⟨fun _ ↦ toPiLocalization_surjective_of_discreteTopology _,
    discreteTopology_of_toLocalization_surjective⟩


theorem discreteTopology_iff_toPiLocalization_bijective {R} [CommSemiring R] :
    DiscreteTopology (PrimeSpectrum R) ↔ Function.Bijective (toPiLocalization R) :=
  discreteTopology_iff_toPiLocalization_surjective.trans
    (and_iff_right <| toPiLocalization_injective _).symm


theorem le_iff_mem_closure (x y : PrimeSpectrum R) :
    x ≤ y ↔ y ∈ closure ({x} : Set (PrimeSpectrum R)) := by
  rw [← asIdeal_le_asIdeal, ← zeroLocus_vanishingIdeal_eq_closure, mem_zeroLocus,
    vanishingIdeal_singleton, SetLike.coe_subset_coe]


theorem le_iff_specializes (x y : PrimeSpectrum R) : x ≤ y ↔ x ⤳ y :=
  (le_iff_mem_closure x y).trans specializes_iff_mem_closure.symm


/-- `nhds` as an order embedding. -/
@[simps!]
def nhdsOrderEmbedding : PrimeSpectrum R ↪o Filter (PrimeSpectrum R) :=
  OrderEmbedding.ofMapLEIff nhds fun a b => (le_iff_specializes a b).symm


instance : T0Space (PrimeSpectrum R) :=
  ⟨nhdsOrderEmbedding.inj'⟩


/-- If `x` specializes to `y`, then there is a natural map from the localization of `y` to the
localization of `x`. -/
def localizationMapOfSpecializes {x y : PrimeSpectrum R} (h : x ⤳ y) :
    Localization.AtPrime y.asIdeal →+* Localization.AtPrime x.asIdeal :=
  @IsLocalization.lift _ _ _ _ _ _ _ _ Localization.isLocalization
    (algebraMap R (Localization.AtPrime x.asIdeal))
    (by
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        x y : PrimeSpectrum R
        h : Specializes x y
        ⊢ ∀ (y_1 : Subtype fun x => Membership.mem y.asIdeal.primeCompl x), IsUnit ((a …
      -/
      rintro ⟨a, ha⟩
      rw [← PrimeSpectrum.le_iff_specializes, ← asIdeal_le_asIdeal, ← SetLike.coe_subset_coe, ←
        Set.compl_subset_compl] at h
      exact (IsLocalization.map_units (Localization.AtPrime x.asIdeal)
        ⟨a, show a ∈ x.asIdeal.primeCompl from h ha⟩ : _))


lemma isClosed_range_of_stableUnderSpecialization
    (hf : StableUnderSpecialization (Set.range (comap f))) :
    IsClosed (Set.range (comap f)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
    ⊢ IsClosed (Set.range ⇑(PrimeSpectrum.comap f))
  -/
  refine (isClosed_iff_zeroLocus _).mpr ⟨RingHom.ker f, le_antisymm ?_ ?_⟩
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
      ⊢ LE.le (Set.range ⇑(PrimeSpectrum.comap f)) (PrimeSpectrum.zeroLocus ↑(RingHo …
    -/
  · rintro _ ⟨q, rfl⟩
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
      q : PrimeSpectrum S
      ⊢ Membership.mem (PrimeSpectrum.zeroLocus ↑(RingHom.ker f)) ((PrimeSpectrum.co …
    -/
    exact Ideal.comap_mono bot_le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
      ⊢ LE.le (PrimeSpectrum.zeroLocus ↑(RingHom.ker f)) (Set.range ⇑(PrimeSpectrum. …
    -/
  · intro p hp
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
      p : PrimeSpectrum R
      hp : Membership.mem (PrimeSpectrum.zeroLocus ↑(RingHom.ker f)) p
      ⊢ Membership.mem (Set.range ⇑(PrimeSpectrum.comap f)) p
    -/
    obtain ⟨q, hq, hqle⟩ := Ideal.exists_minimalPrimes_le hp
    /-
      case refine_2.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
      p : PrimeSpectrum R
      hp : Membership.mem (PrimeSpectrum.zeroLocus ↑(RingHom.ker f)) p
      q : Ideal R
      hq : Membership.mem (RingHom.ker f).minimalPrimes q
      hqle : LE.le q p.asIdeal
      ⊢ Membership.mem (Set.range ⇑(PrimeSpectrum.comap f)) p
    -/
    obtain ⟨q', hq', hq'c⟩ := Ideal.exists_minimalPrimes_comap_eq f q hq
    /-
      case refine_2.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap f))
      p : PrimeSpectrum R
      hp : Membership.mem (PrimeSpectrum.zeroLocus ↑(RingHom.ker f)) p
      q : Ideal R
      hq : Membership.mem (RingHom.ker f).minimalPrimes q
      hqle : LE.le q p.asIdeal
      q' : Ideal S
      hq' : Membership.mem Bot.bot.minimalPrimes q'
      hq'c : Eq (Ideal.comap f q') q
      ⊢ Membership.mem (Set.range ⇑(PrimeSpectrum.comap f)) p
    -/
    exact hf ((le_iff_specializes ⟨q, hq.1.1⟩ p).mp hqle) ⟨⟨q', hq'.1.1⟩, PrimeSpectrum.ext hq'c⟩
    /-
      🎉 no goals
    -/


@[stacks 05JL]
lemma isClosed_image_of_stableUnderSpecialization
    (Z : Set (PrimeSpectrum S)) (hZ : IsClosed Z)
    (hf : StableUnderSpecialization (comap f '' Z)) :
    IsClosed (comap f '' Z) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    Z : Set (PrimeSpectrum S)
    hZ : IsClosed Z
    hf : StableUnderSpecialization (Set.image (⇑(PrimeSpectrum.comap f)) Z)
    ⊢ IsClosed (Set.image (⇑(PrimeSpectrum.comap f)) Z)
  -/
  obtain ⟨I, rfl⟩ := (PrimeSpectrum.isClosed_iff_zeroLocus_ideal Z).mp hZ
  have : (comap f '' zeroLocus I) = Set.range (comap ((Ideal.Quotient.mk I).comp f)) := by
    rw [comap_comp, ContinuousMap.coe_comp, Set.range_comp, range_comap_of_surjective, Ideal.mk_ker]
    exact Ideal.Quotient.mk_surjective
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hZ : IsClosed (PrimeSpectrum.zeroLocus ↑I)
    hf : StableUnderSpecialization (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpe …
    this : Eq (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum.zeroLocus ↑I))  …
    ⊢ IsClosed (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum.zeroLocus ↑I))
  -/
  rw [this] at hf ⊢
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hZ : IsClosed (PrimeSpectrum.zeroLocus ↑I)
    hf : StableUnderSpecialization (Set.range ⇑(PrimeSpectrum.comap ((Ideal.Quotie …
    this : Eq (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum.zeroLocus ↑I))  …
    ⊢ IsClosed (Set.range ⇑(PrimeSpectrum.comap ((Ideal.Quotient.mk I).comp f)))
  -/
  exact isClosed_range_of_stableUnderSpecialization _ hf
  /-
    🎉 no goals
  -/


variable {f} in
@[stacks 05JL]
lemma stableUnderSpecialization_range_iff :
    StableUnderSpecialization (Set.range (comap f)) ↔ IsClosed (Set.range (comap f)) :=
  ⟨isClosed_range_of_stableUnderSpecialization f, fun h ↦ h.stableUnderSpecialization⟩


lemma stableUnderSpecialization_image_iff
    (Z : Set (PrimeSpectrum S)) (hZ : IsClosed Z) :
    StableUnderSpecialization (comap f '' Z) ↔ IsClosed (comap f '' Z) :=
  ⟨isClosed_image_of_stableUnderSpecialization f Z hZ, fun h ↦ h.stableUnderSpecialization⟩


lemma vanishingIdeal_range_comap :
    vanishingIdeal (Set.range (comap f)) = (RingHom.ker f).radical := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (PrimeSpectrum.vanishingIdeal (Set.range ⇑(PrimeSpectrum.comap f))) (Ring …
  -/
  ext x
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    x : R
    ⊢ Iff (Membership.mem (PrimeSpectrum.vanishingIdeal (Set.range ⇑(PrimeSpectrum …
  -/
  rw [RingHom.ker_eq_comap_bot, ← Ideal.comap_radical, Ideal.radical_eq_sInf]
  simp only [mem_vanishingIdeal, Set.mem_range, forall_exists_index, forall_apply_eq_imp_iff,
    comap_asIdeal, Ideal.mem_comap, bot_le, true_and, Submodule.mem_sInf, Set.mem_setOf_eq]
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    x : R
    ⊢ Iff (∀ (a : PrimeSpectrum S), Membership.mem a.asIdeal (f x)) (∀ (p : Submod …
  -/
  exact ⟨fun H I hI ↦ H ⟨I, hI⟩, fun H I ↦ H I.1 I.2⟩
  /-
    🎉 no goals
  -/


lemma closure_range_comap :
    closure (Set.range (comap f)) = zeroLocus (RingHom.ker f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (closure (Set.range ⇑(PrimeSpectrum.comap f))) (PrimeSpectrum.zeroLocus ↑ …
  -/
  rw [← zeroLocus_vanishingIdeal_eq_closure, vanishingIdeal_range_comap, zeroLocus_radical]
  /-
    🎉 no goals
  -/


lemma denseRange_comap_iff_ker_le_nilRadical :
    DenseRange (comap f) ↔ RingHom.ker f ≤ nilradical R := by
  rw [denseRange_iff_closure_range, closure_range_comap, ← Set.top_eq_univ, zeroLocus_eq_top_iff,
    SetLike.coe_subset_coe]


@[stacks 00FL]
lemma denseRange_comap_iff_minimalPrimes :
    DenseRange (comap f) ↔ ∀ I (h : I ∈ minimalPrimes R), ⟨I, h.1.1⟩ ∈ Set.range (comap f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Iff (DenseRange ⇑(PrimeSpectrum.comap f)) (∀ (I : Ideal R) (h : Membership.m …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      ⊢ DenseRange ⇑(PrimeSpectrum.comap f) → ∀ (I : Ideal R) (h : Membership.mem (m …
    -/
  · intro H I hI
    have : I ∈ (RingHom.ker f).minimalPrimes := by
      rw [denseRange_comap_iff_ker_le_nilRadical] at H
      simp only [minimalPrimes, Ideal.minimalPrimes, Set.mem_setOf] at hI ⊢
      convert hI using 2 with p
      exact ⟨fun h ↦ ⟨h.1, bot_le⟩, fun h ↦ ⟨h.1, H.trans (h.1.radical_le_iff.mpr bot_le)⟩⟩
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      H : DenseRange ⇑(PrimeSpectrum.comap f)
      I : Ideal R
      hI : Membership.mem (minimalPrimes R) I
      this : Membership.mem (RingHom.ker f).minimalPrimes I
      ⊢ Membership.mem (Set.range ⇑(PrimeSpectrum.comap f)) { asIdeal := I, isPrime  …
    -/
    obtain ⟨p, hp, _, rfl⟩ := Ideal.exists_comap_eq_of_mem_minimalPrimes f (I := ⊥) I this
    /-
      case mp.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      H : DenseRange ⇑(PrimeSpectrum.comap f)
      p : Ideal S
      hp : p.IsPrime
      left✝ : LE.le Bot.bot p
      hI : Membership.mem (minimalPrimes R) (Ideal.comap f p)
      this : Membership.mem (RingHom.ker f).minimalPrimes (Ideal.comap f p)
      ⊢ Membership.mem (Set.range ⇑(PrimeSpectrum.comap f)) { asIdeal := Ideal.comap …
    -/
    exact ⟨⟨p, hp⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      ⊢ (∀ (I : Ideal R) (h : Membership.mem (minimalPrimes R) I), Membership.mem (S …
    -/
  · intro H p
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      H : ∀ (I : Ideal R) (h : Membership.mem (minimalPrimes R) I), Membership.mem ( …
      p : PrimeSpectrum R
      ⊢ Membership.mem (closure (Set.range ⇑(PrimeSpectrum.comap f))) p
    -/
    obtain ⟨q, hq, hq'⟩ := Ideal.exists_minimalPrimes_le (J := p.asIdeal) bot_le
    exact ((le_iff_specializes ⟨q, hq.1.1⟩ p).mp hq').mem_closed isClosed_closure
      (subset_closure (H q hq))


variable (R) in
/--
Zero loci of prime ideals are closed irreducible sets in the Zariski topology and any closed
irreducible set is a zero locus of some prime ideal.
-/
protected def pointsEquivIrreducibleCloseds :
    PrimeSpectrum R ≃o (TopologicalSpace.IrreducibleCloseds (PrimeSpectrum R))ᵒᵈ where
  __ := irreducibleSetEquivPoints.toEquiv.symm.trans OrderDual.toDual
  map_rel_iff' {p q} :=
    (RelIso.symm irreducibleSetEquivPoints).map_rel_iff.trans (le_iff_specializes p q).symm


/-- Also see `PrimeSpectrum.isClosed_singleton_iff_isMaximal` -/
lemma isMax_iff {x : PrimeSpectrum R} :
    IsMax x ↔ x.asIdeal.IsMaximal := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (IsMax x) x.asIdeal.IsMaximal
  -/
  refine ⟨fun hx ↦ ⟨⟨x.2.ne_top, fun I hI ↦ ?_⟩⟩, fun hx y e ↦ (hx.eq_of_le y.2.ne_top e).ge⟩
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    hx : IsMax x
    I : Ideal R
    hI : LT.lt x.asIdeal I
    ⊢ Eq I Top.top
  -/
  by_contra e
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    hx : IsMax x
    I : Ideal R
    hI : LT.lt x.asIdeal I
    e : Not (Eq I Top.top)
    ⊢ False
  -/
  obtain ⟨m, hm, hm'⟩ := Ideal.exists_le_maximal I e
  /-
    case intro.intro
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    hx : IsMax x
    I : Ideal R
    hI : LT.lt x.asIdeal I
    e : Not (Eq I Top.top)
    m : Ideal R
    hm : m.IsMaximal
    hm' : LE.le I m
    ⊢ False
  -/
  exact hx.not_lt (show x < ⟨m, hm.isPrime⟩ from hI.trans_le hm')
  /-
    🎉 no goals
  -/


lemma stableUnderSpecialization_singleton {x : PrimeSpectrum R} :
    StableUnderSpecialization {x} ↔ x.asIdeal.IsMaximal := by
  simp_rw [← isMax_iff, StableUnderSpecialization, ← le_iff_specializes, Set.mem_singleton_iff,
    @forall_comm _ (_ = _), forall_eq]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (∀ (a : PrimeSpectrum R), LE.le x a → Eq a x) (IsMax x)
  -/
  exact ⟨fun H a h ↦ (H a h).le, fun H a h ↦ le_antisymm (H h) h⟩
  /-
    🎉 no goals
  -/


lemma isMin_iff {x : PrimeSpectrum R} :
    IsMin x ↔ x.asIdeal ∈ minimalPrimes R := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (IsMin x) (Membership.mem (minimalPrimes R) x.asIdeal)
  -/
  show IsMin _ ↔ Minimal (fun q : Ideal R ↦ q.IsPrime ∧ ⊥ ≤ q) _
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (IsMin x) (Minimal (fun q => And q.IsPrime (LE.le Bot.bot q)) x.asIdeal)
  -/
  simp only [IsMin, Minimal, x.2, bot_le, and_self, and_true, true_and]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (∀ ⦃b : PrimeSpectrum R⦄, LE.le b x → LE.le x b) (∀ ⦃y : Ideal R⦄, y.IsP …
  -/
  exact ⟨fun H y hy e ↦ @H ⟨y, hy⟩ e, fun H y e ↦ H y.2 e⟩
  /-
    🎉 no goals
  -/


lemma stableUnderGeneralization_singleton {x : PrimeSpectrum R} :
    StableUnderGeneralization {x} ↔ x.asIdeal ∈ minimalPrimes R := by
  simp_rw [← isMin_iff, StableUnderGeneralization, ← le_iff_specializes, Set.mem_singleton_iff,
    @forall_comm _ (_ = _), forall_eq]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : PrimeSpectrum R
    ⊢ Iff (∀ (a : PrimeSpectrum R), LE.le a x → Eq a x) (IsMin x)
  -/
  exact ⟨fun H a h ↦ (H a h).ge, fun H a h ↦ le_antisymm h (H h)⟩
  /-
    🎉 no goals
  -/


lemma isCompact_isOpen_iff {s : Set (PrimeSpectrum R)} :
    IsCompact s ∧ IsOpen s ↔ ∃ t : Finset R, (zeroLocus t)ᶜ = s := by
  rw [isCompact_open_iff_eq_finite_iUnion_of_isTopologicalBasis _
    isTopologicalBasis_basic_opens isCompact_basicOpen]
  simp only [basicOpen_eq_zeroLocus_compl, ← Set.compl_iInter₂, ← zeroLocus_iUnion₂,
    Set.biUnion_of_singleton]
  exact ⟨fun ⟨s, hs, e⟩ ↦ ⟨hs.toFinset, by simpa using e.symm⟩,
    fun ⟨s, e⟩ ↦ ⟨s, s.finite_toSet, by simpa using e.symm⟩⟩


lemma isCompact_isOpen_iff_ideal {s : Set (PrimeSpectrum R)} :
    IsCompact s ∧ IsOpen s ↔ ∃ I : Ideal R, I.FG ∧ (zeroLocus I)ᶜ = s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    ⊢ Iff (And (IsCompact s) (IsOpen s)) (Exists fun I => And I.FG (Eq (HasCompl.c …
  -/
  rw [isCompact_isOpen_iff]
  exact ⟨fun ⟨s, e⟩ ↦ ⟨.span s, ⟨s, rfl⟩, by simpa using e⟩,
    fun ⟨I, ⟨s, hs⟩, e⟩ ↦ ⟨s, by simpa [hs.symm] using e⟩⟩


lemma basicOpen_injOn_isIdempotentElem :
    {e : R | IsIdempotentElem e}.InjOn basicOpen := fun x hx y hy eq ↦ by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : R
    hx : Membership.mem (setOf fun e => IsIdempotentElem e) x
    y : R
    hy : Membership.mem (setOf fun e => IsIdempotentElem e) y
    eq : Eq (PrimeSpectrum.basicOpen x) (PrimeSpectrum.basicOpen y)
    ⊢ Eq x y
  -/
  by_contra! ne
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : R
    hx : Membership.mem (setOf fun e => IsIdempotentElem e) x
    y : R
    hy : Membership.mem (setOf fun e => IsIdempotentElem e) y
    eq : Eq (PrimeSpectrum.basicOpen x) (PrimeSpectrum.basicOpen y)
    ne : Ne x y
    ⊢ False
  -/
  wlog ne' : x * y ≠ x generalizing x y
    /-
      case inr
      R : Type u
      inst✝ : CommSemiring R
      x : R
      hx : Membership.mem (setOf fun e => IsIdempotentElem e) x
      y : R
      hy : Membership.mem (setOf fun e => IsIdempotentElem e) y
      eq : Eq (PrimeSpectrum.basicOpen x) (PrimeSpectrum.basicOpen y)
      ne : Ne x y
      this : ∀ (x : R), Membership.mem (setOf fun e => IsIdempotentElem e) x → ∀ (y  …
      ne' : Not (Ne (HMul.hMul x y) x)
      ⊢ False
    -/
  · apply this y hy x hx eq.symm ne.symm
    /-
      case inr
      R : Type u
      inst✝ : CommSemiring R
      x : R
      hx : Membership.mem (setOf fun e => IsIdempotentElem e) x
      y : R
      hy : Membership.mem (setOf fun e => IsIdempotentElem e) y
      eq : Eq (PrimeSpectrum.basicOpen x) (PrimeSpectrum.basicOpen y)
      ne : Ne x y
      this : ∀ (x : R), Membership.mem (setOf fun e => IsIdempotentElem e) x → ∀ (y  …
      ne' : Not (Ne (HMul.hMul x y) x)
      ⊢ Ne (HMul.hMul y x) y
    -/
    rwa [mul_comm, of_not_not ne']
    /-
      🎉 no goals
    -/
  have : x ∉ Ideal.span {y} := fun mem ↦ ne' <| by
    obtain ⟨r, rfl⟩ := Ideal.mem_span_singleton'.mp mem
    rw [mul_assoc, hy]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x : R
    hx : Membership.mem (setOf fun e => IsIdempotentElem e) x
    y : R
    hy : Membership.mem (setOf fun e => IsIdempotentElem e) y
    eq : Eq (PrimeSpectrum.basicOpen x) (PrimeSpectrum.basicOpen y)
    ne : Ne x y
    ne' : Ne (HMul.hMul x y) x
    this : Not (Membership.mem (Ideal.span (Singleton.singleton y)) x)
    ⊢ False
  -/
  have ⟨p, prime, le, nmem⟩ := Ideal.exists_le_prime_nmem_of_isIdempotentElem _ x hx this
  exact ne_of_mem_of_not_mem' (a := ⟨p, prime⟩) nmem
    (not_not.mpr <| p.span_singleton_le_iff_mem.mp le) eq


@[stacks 00EE]
lemma existsUnique_idempotent_basicOpen_eq_of_isClopen {s : Set (PrimeSpectrum R)}
    (hs : IsClopen s) : ∃! e : R, IsIdempotentElem e ∧ s = basicOpen e := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    ⊢ ExistsUnique fun e => And (IsIdempotentElem e) (Eq s ↑(PrimeSpectrum.basicOp …
  -/
  refine existsUnique_of_exists_of_unique ?_ ?_; swap
    /-
      case refine_2
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      hs : IsClopen s
      ⊢ ∀ (y₁ y₂ : R), And (IsIdempotentElem y₁) (Eq s ↑(PrimeSpectrum.basicOpen y₁) …
    -/
  · rintro x y ⟨hx, rfl⟩ ⟨hy, eq⟩
    /-
      case refine_2.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      x y : R
      hx : IsIdempotentElem x
      hs : IsClopen ↑(PrimeSpectrum.basicOpen x)
      hy : IsIdempotentElem y
      eq : Eq ↑(PrimeSpectrum.basicOpen x) ↑(PrimeSpectrum.basicOpen y)
      ⊢ Eq x y
    -/
    exact basicOpen_injOn_isIdempotentElem hx hy (SetLike.ext' eq)
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  cases subsingleton_or_nontrivial R
    /-
      case refine_1.inl
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      hs : IsClopen s
      h✝ : Subsingleton R
      ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
    -/
  · exact ⟨0, Subsingleton.elim _ _, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_1.inr
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  obtain ⟨I, hI, hI'⟩ := isCompact_isOpen_iff_ideal.mp ⟨hs.1.isCompact, hs.2⟩
  obtain ⟨J, hJ, hJ'⟩ := isCompact_isOpen_iff_ideal.mp
    ⟨hs.2.isClosed_compl.isCompact, hs.1.isOpen_compl⟩
  /-
    case refine_1.inr.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    hI' : Eq (HasCompl.compl (PrimeSpectrum.zeroLocus ↑I)) s
    J : Ideal R
    hJ : J.FG
    hJ' : Eq (HasCompl.compl (PrimeSpectrum.zeroLocus ↑J)) (HasCompl.compl s)
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  simp only [compl_eq_iff_isCompl, ← eq_compl_iff_isCompl, compl_compl] at hI' hJ'
  have : I * J ≤ nilradical R := by
    refine Ideal.radical_le_radical_iff.mp (le_of_eq ?_)
    rw [← zeroLocus_eq_iff, Ideal.zero_eq_bot, zeroLocus_bot,
      zeroLocus_mul, hI', hJ', Set.compl_union_self]
  /-
    case refine_1.inr.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    J : Ideal R
    hJ : J.FG
    hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
    hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
    this : LE.le (HMul.hMul I J) (nilradical R)
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  obtain ⟨n, hn⟩ := Ideal.exists_pow_le_of_le_radical_of_fg this (Submodule.FG.mul hI hJ)
  /-
    case refine_1.inr.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    J : Ideal R
    hJ : J.FG
    hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
    hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
    this : LE.le (HMul.hMul I J) (nilradical R)
    n : Nat
    hn : LE.le (HPow.hPow (HMul.hMul I J) n) 0
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  have hnz : n ≠ 0 := by rintro rfl; simp at hn
  /-
    case refine_1.inr.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    J : Ideal R
    hJ : J.FG
    hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
    hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
    this : LE.le (HMul.hMul I J) (nilradical R)
    n : Nat
    hn : LE.le (HPow.hPow (HMul.hMul I J) n) 0
    hnz : Ne n 0
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  rw [mul_pow, Ideal.zero_eq_bot] at hn
  have : I ^ n ⊔ J ^ n = ⊤ := by
    rw [eq_top_iff, ← Ideal.span_pow_eq_top (I ∪ J : Set R) _ n, Ideal.span_le, Set.image_union,
      Set.union_subset_iff]
    constructor
    · rintro _ ⟨x, hx, rfl⟩; exact Ideal.mem_sup_left (Ideal.pow_mem_pow hx n)
    · rintro _ ⟨x, hx, rfl⟩; exact Ideal.mem_sup_right (Ideal.pow_mem_pow hx n)
    · rw [Ideal.span_union, Ideal.span_eq, Ideal.span_eq, ← zeroLocus_empty_iff_eq_top,
        zeroLocus_sup, hI', hJ', Set.compl_inter_self]
  /-
    case refine_1.inr.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    J : Ideal R
    hJ : J.FG
    hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
    hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
    this✝ : LE.le (HMul.hMul I J) (nilradical R)
    n : Nat
    hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
    hnz : Ne n 0
    this : Eq (Max.max (HPow.hPow I n) (HPow.hPow J n)) Top.top
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  rw [Ideal.eq_top_iff_one, Submodule.mem_sup] at this
  /-
    case refine_1.inr.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    J : Ideal R
    hJ : J.FG
    hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
    hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
    this✝ : LE.le (HMul.hMul I J) (nilradical R)
    n : Nat
    hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
    hnz : Ne n 0
    this : Exists fun y => And (Membership.mem (HPow.hPow I n) y) (Exists fun z => …
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  obtain ⟨x, hx, y, hy, e⟩ := this
  /-
    case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    hs : IsClopen s
    h✝ : Nontrivial R
    I : Ideal R
    hI : I.FG
    J : Ideal R
    hJ : J.FG
    hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
    hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
    this : LE.le (HMul.hMul I J) (nilradical R)
    n : Nat
    hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
    hnz : Ne n 0
    x : R
    hx : Membership.mem (HPow.hPow I n) x
    y : R
    hy : Membership.mem (HPow.hPow J n) y
    e : Eq (HAdd.hAdd x y) 1
    ⊢ Exists fun x => And (IsIdempotentElem x) (Eq s ↑(PrimeSpectrum.basicOpen x))
  -/
  refine ⟨x, ?_, subset_antisymm ?_ ?_⟩
    /-
      case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      hs : IsClopen s
      h✝ : Nontrivial R
      I : Ideal R
      hI : I.FG
      J : Ideal R
      hJ : J.FG
      hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
      hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
      this : LE.le (HMul.hMul I J) (nilradical R)
      n : Nat
      hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
      hnz : Ne n 0
      x : R
      hx : Membership.mem (HPow.hPow I n) x
      y : R
      hy : Membership.mem (HPow.hPow J n) y
      e : Eq (HAdd.hAdd x y) 1
      ⊢ IsIdempotentElem x
    -/
  · replace e := congr(x * $e)
    /-
      case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      hs : IsClopen s
      h✝ : Nontrivial R
      I : Ideal R
      hI : I.FG
      J : Ideal R
      hJ : J.FG
      hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
      hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
      this : LE.le (HMul.hMul I J) (nilradical R)
      n : Nat
      hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
      hnz : Ne n 0
      x : R
      hx : Membership.mem (HPow.hPow I n) x
      y : R
      hy : Membership.mem (HPow.hPow J n) y
      e : Eq (HMul.hMul x (HAdd.hAdd x y)) (HMul.hMul x 1)
      ⊢ IsIdempotentElem x
    -/
    rwa [mul_add, hn (Ideal.mul_mem_mul hx hy), add_zero, mul_one] at e
    /-
      🎉 no goals
    -/
  · rw [PrimeSpectrum.basicOpen_eq_zeroLocus_compl, Set.subset_compl_iff_disjoint_left,
      Set.disjoint_iff_inter_eq_empty, ← hJ', ← zeroLocus_span,
      ← zeroLocus_sup, zeroLocus_empty_iff_eq_top,
      Ideal.eq_top_iff_one, ← e]
    /-
      case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      hs : IsClopen s
      h✝ : Nontrivial R
      I : Ideal R
      hI : I.FG
      J : Ideal R
      hJ : J.FG
      hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
      hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
      this : LE.le (HMul.hMul I J) (nilradical R)
      n : Nat
      hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
      hnz : Ne n 0
      x : R
      hx : Membership.mem (HPow.hPow I n) x
      y : R
      hy : Membership.mem (HPow.hPow J n) y
      e : Eq (HAdd.hAdd x y) 1
      ⊢ Membership.mem (Max.max (Ideal.span (Singleton.singleton x)) J) (HAdd.hAdd x …
    -/
    exact Submodule.add_mem_sup (Ideal.subset_span (Set.mem_singleton _)) (Ideal.pow_le_self hnz hy)
    /-
      🎉 no goals
    -/
    /-
      case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      hs : IsClopen s
      h✝ : Nontrivial R
      I : Ideal R
      hI : I.FG
      J : Ideal R
      hJ : J.FG
      hI' : Eq (PrimeSpectrum.zeroLocus ↑I) (HasCompl.compl s)
      hJ' : Eq (PrimeSpectrum.zeroLocus ↑J) s
      this : LE.le (HMul.hMul I J) (nilradical R)
      n : Nat
      hn : LE.le (HMul.hMul (HPow.hPow I n) (HPow.hPow J n)) Bot.bot
      hnz : Ne n 0
      x : R
      hx : Membership.mem (HPow.hPow I n) x
      y : R
      hy : Membership.mem (HPow.hPow J n) y
      e : Eq (HAdd.hAdd x y) 1
      ⊢ HasSubset.Subset (↑(PrimeSpectrum.basicOpen x)) s
    -/
  · rw [PrimeSpectrum.basicOpen_eq_zeroLocus_compl, Set.compl_subset_comm, ← hI']
    exact PrimeSpectrum.zeroLocus_anti_mono
      (Set.singleton_subset_iff.mpr <| Ideal.pow_le_self hnz hx)


lemma exists_idempotent_basicOpen_eq_of_isClopen {s : Set (PrimeSpectrum R)}
    (hs : IsClopen s) : ∃ e : R, IsIdempotentElem e ∧ s = basicOpen e :=
  (existsUnique_idempotent_basicOpen_eq_of_isClopen hs).exists


@[deprecated (since := "2024-11-11")]
alias exists_idempotent_basicOpen_eq_of_is_clopen := exists_idempotent_basicOpen_eq_of_isClopen


theorem isClosedMap_comap_of_isIntegral (hf : f.IsIntegral) :
    IsClosedMap (comap f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : f.IsIntegral
    ⊢ IsClosedMap ⇑(PrimeSpectrum.comap f)
  -/
  refine fun s hs ↦ isClosed_image_of_stableUnderSpecialization _ _ hs ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : f.IsIntegral
    s : Set (PrimeSpectrum S)
    hs : IsClosed s
    ⊢ StableUnderSpecialization (Set.image (⇑(PrimeSpectrum.comap f)) s)
  -/
  rintro _ y e ⟨x, hx, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : f.IsIntegral
    s : Set (PrimeSpectrum S)
    hs : IsClosed s
    y : PrimeSpectrum R
    x : PrimeSpectrum S
    hx : Membership.mem s x
    e : Specializes ((PrimeSpectrum.comap f) x) y
    ⊢ Membership.mem (Set.image (⇑(PrimeSpectrum.comap f)) s) y
  -/
  algebraize [f]
  obtain ⟨q, hq₁, hq₂, hq₃⟩ := Ideal.exists_ideal_over_prime_of_isIntegral y.asIdeal x.asIdeal
    ((le_iff_specializes _ _).mpr e)
  refine ⟨⟨q, hq₂⟩, ((le_iff_specializes _ ⟨q, hq₂⟩).mp hq₁).mem_closed hs hx,
    PrimeSpectrum.ext hq₃⟩


theorem isClosed_comap_singleton_of_isIntegral (hf : f.IsIntegral)
    (x : PrimeSpectrum S) (hx : IsClosed ({x} : Set (PrimeSpectrum S))) :
    IsClosed ({comap f x} : Set (PrimeSpectrum R)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : f.IsIntegral
    x : PrimeSpectrum S
    hx : IsClosed (Singleton.singleton x)
    ⊢ IsClosed (Singleton.singleton ((PrimeSpectrum.comap f) x))
  -/
  simpa using isClosedMap_comap_of_isIntegral f hf _ hx
  /-
    🎉 no goals
  -/


lemma closure_image_comap_zeroLocus (I : Ideal S) :
    closure (comap f '' zeroLocus I) = zeroLocus (I.comap f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    ⊢ Eq (closure (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum.zeroLocus ↑ …
  -/
  apply subset_antisymm
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      ⊢ HasSubset.Subset (closure (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectr …
    -/
  · rw [(isClosed_zeroLocus _).closure_subset_iff, Set.image_subset_iff, preimage_comap_zeroLocus]
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (PrimeSpectrum.zeroLocus (Set. …
    -/
    exact zeroLocus_anti_mono (Set.image_preimage_subset _ _)
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus ↑(Ideal.comap f I)) (closure (Set. …
    -/
  · rintro x (hx : I.comap f ≤ x.asIdeal)
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      x : PrimeSpectrum R
      hx : LE.le (Ideal.comap f I) x.asIdeal
      ⊢ Membership.mem (closure (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum …
    -/
    obtain ⟨q, hq₁, hq₂⟩ := Ideal.exists_minimalPrimes_le hx
    /-
      case a.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      x : PrimeSpectrum R
      hx : LE.le (Ideal.comap f I) x.asIdeal
      q : Ideal R
      hq₁ : Membership.mem (Ideal.comap f I).minimalPrimes q
      hq₂ : LE.le q x.asIdeal
      ⊢ Membership.mem (closure (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum …
    -/
    obtain ⟨p', hp', hp'', rfl⟩ := Ideal.exists_comap_eq_of_mem_minimalPrimes f _ hq₁
    /-
      case a.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      I : Ideal S
      x : PrimeSpectrum R
      hx : LE.le (Ideal.comap f I) x.asIdeal
      p' : Ideal S
      hp' : p'.IsPrime
      hp'' : LE.le I p'
      hq₁ : Membership.mem (Ideal.comap f I).minimalPrimes (Ideal.comap f p')
      hq₂ : LE.le (Ideal.comap f p') x.asIdeal
      ⊢ Membership.mem (closure (Set.image (⇑(PrimeSpectrum.comap f)) (PrimeSpectrum …
    -/
    let p'' : PrimeSpectrum S := ⟨p', hp'⟩
    apply isClosed_closure.stableUnderSpecialization ((le_iff_specializes
      (comap f ⟨p', hp'⟩) x).mp hq₂) (subset_closure (by exact ⟨_, hp'', rfl⟩))


lemma isIntegral_of_isClosedMap_comap_mapRingHom (h : IsClosedMap (comap (mapRingHom f))) :
    f.IsIntegral := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    ⊢ f.IsIntegral
  -/
  algebraize [f]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    ⊢ f.IsIntegral
  -/
  suffices Algebra.IsIntegral R S by rwa [Algebra.isIntegral_def] at this
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    ⊢ Algebra.IsIntegral R S
  -/
  nontriviality R
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝ : Nontrivial R
    ⊢ Algebra.IsIntegral R S
  -/
  nontriviality S
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝¹ : Nontrivial R
    a✝ : Nontrivial S
    ⊢ Algebra.IsIntegral R S
  -/
  constructor
  /-
    case isIntegral
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝¹ : Nontrivial R
    a✝ : Nontrivial S
    ⊢ ∀ (x : S), IsIntegral R x
  -/
  intro r
  /-
    case isIntegral
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝¹ : Nontrivial R
    a✝ : Nontrivial S
    r : S
    ⊢ IsIntegral R r
  -/
  let p : S[X] := C r * X - 1
  have : (1 : R[X]) ∈ Ideal.span {X} ⊔ (Ideal.span {p}).comap (mapRingHom f) := by
    have H := h _ (isClosed_zeroLocus {p})
    rw [← zeroLocus_span, ← closure_eq_iff_isClosed, closure_image_comap_zeroLocus] at H
    rw [← Ideal.eq_top_iff_one, sup_comm, ← zeroLocus_empty_iff_eq_top, zeroLocus_sup, H]
    suffices ∀ (a : PrimeSpectrum S[X]), p ∈ a.asIdeal → X ∉ a.asIdeal by
      simpa [Set.eq_empty_iff_forall_not_mem]
    intro q hpq hXq
    have : 1 ∈ q.asIdeal := by simpa [p] using (sub_mem (q.asIdeal.mul_mem_left (C r) hXq) hpq)
    exact q.2.ne_top (q.asIdeal.eq_top_iff_one.mpr this)
  /-
    case isIntegral
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝¹ : Nontrivial R
    a✝ : Nontrivial S
    r : S
    p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
    this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
    ⊢ IsIntegral R r
  -/
  obtain ⟨a, b, hb, e⟩ := Ideal.mem_span_singleton_sup.mp this
  /-
    case isIntegral.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝¹ : Nontrivial R
    a✝ : Nontrivial S
    r : S
    p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
    this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
    a b : Polynomial R
    hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
    e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
    ⊢ IsIntegral R r
  -/
  obtain ⟨c, hc : b.map (algebraMap R S) = _⟩ := Ideal.mem_span_singleton.mp hb
  /-
    case isIntegral.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
    algInst✝ : Algebra R S := f.toAlgebra
    a✝¹ : Nontrivial R
    a✝ : Nontrivial S
    r : S
    p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
    this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
    a b : Polynomial R
    hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
    e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
    c : Polynomial S
    hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
    ⊢ IsIntegral R r
  -/
  refine ⟨b.reverse * X ^ (1 + c.natDegree), ?_, ?_⟩
    /-
      case isIntegral.intro.intro.intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
      algInst✝ : Algebra R S := f.toAlgebra
      a✝¹ : Nontrivial R
      a✝ : Nontrivial S
      r : S
      p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
      this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
      a b : Polynomial R
      hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
      e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
      c : Polynomial S
      hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
      ⊢ (HMul.hMul b.reverse (HPow.hPow Polynomial.X (HAdd.hAdd 1 c.natDegree))).Monic
    -/
  · refine Monic.mul ?_ (by simp)
    /-
      case isIntegral.intro.intro.intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
      algInst✝ : Algebra R S := f.toAlgebra
      a✝¹ : Nontrivial R
      a✝ : Nontrivial S
      r : S
      p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
      this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
      a b : Polynomial R
      hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
      e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
      c : Polynomial S
      hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
      ⊢ b.reverse.Monic
    -/
    have h : b.coeff 0 = 1 := by simpa using congr(($e).coeff 0)
    /-
      case isIntegral.intro.intro.intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      h✝ : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
      algInst✝ : Algebra R S := f.toAlgebra
      a✝¹ : Nontrivial R
      a✝ : Nontrivial S
      r : S
      p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
      this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
      a b : Polynomial R
      hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
      e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
      c : Polynomial S
      hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
      h : Eq (b.coeff 0) 1
      ⊢ b.reverse.Monic
    -/
    have : b.natTrailingDegree = 0 := by simp [h]
    /-
      case isIntegral.intro.intro.intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      h✝ : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
      algInst✝ : Algebra R S := f.toAlgebra
      a✝¹ : Nontrivial R
      a✝ : Nontrivial S
      r : S
      p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
      this✝ : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X) …
      a b : Polynomial R
      hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
      e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
      c : Polynomial S
      hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
      h : Eq (b.coeff 0) 1
      this : Eq b.natTrailingDegree 0
      ⊢ b.reverse.Monic
    -/
    rw [Monic.def, reverse_leadingCoeff, trailingCoeff, this, h]
    /-
      🎉 no goals
    -/
    /-
      case isIntegral.intro.intro.intro.intro.refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
      algInst✝ : Algebra R S := f.toAlgebra
      a✝¹ : Nontrivial R
      a✝ : Nontrivial S
      r : S
      p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
      this : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X)) …
      a b : Polynomial R
      hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
      e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
      c : Polynomial S
      hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
      ⊢ Eq (Polynomial.eval₂ (algebraMap R S) r (HMul.hMul b.reverse (HPow.hPow Poly …
    -/
  · have : p.natDegree ≤ 1 := by simpa using natDegree_linear_le (a := r) (b := -1)
    rw [eval₂_eq_eval_map, reverse, Polynomial.map_mul, ← reflect_map, Polynomial.map_pow,
      map_X, ← revAt_zero (1 + _), ← reflect_monomial,
      ← reflect_mul _ _ natDegree_map_le (by simp), pow_zero, mul_one, hc,
      ← add_assoc, reflect_mul _ _ (this.trans (by simp)) le_rfl,
      eval_mul, reflect_sub, reflect_mul _ _ (by simp) (by simp)]
    /-
      case isIntegral.intro.intro.intro.intro.refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      h : IsClosedMap ⇑(PrimeSpectrum.comap (Polynomial.mapRingHom f))
      algInst✝ : Algebra R S := f.toAlgebra
      a✝¹ : Nontrivial R
      a✝ : Nontrivial S
      r : S
      p : Polynomial S := HSub.hSub (HMul.hMul (Polynomial.C r) Polynomial.X) 1
      this✝ : Membership.mem (Max.max (Ideal.span (Singleton.singleton Polynomial.X) …
      a b : Polynomial R
      hb : Membership.mem (Ideal.comap (Polynomial.mapRingHom f) (Ideal.span (Single …
      e : Eq (HAdd.hAdd (HMul.hMul a Polynomial.X) b) 1
      c : Polynomial S
      hc : Eq (Polynomial.map (algebraMap R S) b) (HMul.hMul p c)
      this : LE.le p.natDegree 1
      ⊢ Eq (HMul.hMul (Polynomial.eval r (HSub.hSub (HMul.hMul (Polynomial.reflect b …
    -/
    simp [← pow_succ']
    /-
      🎉 no goals
    -/


/--
Localizations at minimal primes have single-point prime spectra.
-/
def primeSpectrum_unique_of_localization_at_minimal (h : I ∈ minimalPrimes R) :
    Unique (PrimeSpectrum (Localization.AtPrime I)) where
  default :=
    ⟨IsLocalRing.maximalIdeal (Localization I.primeCompl),
    (IsLocalRing.maximalIdeal.isMaximal _).isPrime⟩
  uniq x := PrimeSpectrum.ext (Localization.AtPrime.prime_unique_of_minimal h x.asIdeal)


open PrimeSpectrum in
/--
[Stacks: Lemma 00ES (3)](https://stacks.math.columbia.edu/tag/00ES)
Zero loci of minimal prime ideals of `R` are irreducible components in `Spec R` and any
irreducible component is a zero locus of some minimal prime ideal.
-/
protected def minimalPrimes.equivIrreducibleComponents :
    minimalPrimes R ≃o (irreducibleComponents <| PrimeSpectrum R)ᵒᵈ := by
  let e : {p : Ideal R | p.IsPrime ∧ ⊥ ≤ p} ≃o PrimeSpectrum R :=
    ⟨⟨fun x ↦ ⟨x.1, x.2.1⟩, fun x ↦ ⟨x.1, x.2, bot_le⟩, fun _ ↦ rfl, fun _ ↦ rfl⟩, Iff.rfl⟩
  /-
    R : Type u
    S : Type v
    inst✝ : CommSemiring R
    e : OrderIso (↑(setOf fun p => And p.IsPrime (LE.le Bot.bot p))) (PrimeSpectru …
    ⊢ OrderIso (↑(minimalPrimes R)) (OrderDual ↑(irreducibleComponents (PrimeSpect …
  -/
  rw [irreducibleComponents_eq_maximals_closed]
  exact OrderIso.setOfMinimalIsoSetOfMaximal
    (e.trans ((PrimeSpectrum.pointsEquivIrreducibleCloseds R).trans
    (TopologicalSpace.IrreducibleCloseds.orderIsoSubtype' (PrimeSpectrum R)).dual))


lemma vanishingIdeal_irreducibleComponents :
    vanishingIdeal '' (irreducibleComponents <| PrimeSpectrum R) =
    minimalPrimes R := by
  rw [irreducibleComponents_eq_maximals_closed, minimalPrimes_eq_minimals,
    image_antitone_setOf_maximal (fun s t hs _ ↦ (vanishingIdeal_anti_mono_iff hs.1).symm),
    ← funext (@Set.mem_setOf_eq _ · Ideal.IsPrime), ← vanishingIdeal_isClosed_isIrreducible]
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (setOf fun x => Minimal (fun x => Exists fun x₀ => And (And (IsClosed x₀) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma zeroLocus_minimalPrimes :
    zeroLocus ∘ (↑) '' minimalPrimes R =
    irreducibleComponents (PrimeSpectrum R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (Set.image (Function.comp PrimeSpectrum.zeroLocus SetLike.coe) (minimalPr …
  -/
  rw [← vanishingIdeal_irreducibleComponents, ← Set.image_comp, Set.EqOn.image_eq_self]
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Set.EqOn (Function.comp (Function.comp PrimeSpectrum.zeroLocus SetLike.coe)  …
  -/
  intros s hs
  simpa [zeroLocus_vanishingIdeal_eq_closure, closure_eq_iff_isClosed]
    using isClosed_of_mem_irreducibleComponents s hs


lemma vanishingIdeal_mem_minimalPrimes {s : Set (PrimeSpectrum R)} :
    vanishingIdeal s ∈ minimalPrimes R ↔ closure s ∈ irreducibleComponents (PrimeSpectrum R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set (PrimeSpectrum R)
    ⊢ Iff (Membership.mem (minimalPrimes R) (PrimeSpectrum.vanishingIdeal s)) (Mem …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      ⊢ Membership.mem (minimalPrimes R) (PrimeSpectrum.vanishingIdeal s) → Membersh …
    -/
  · rw [← zeroLocus_minimalPrimes, ← zeroLocus_vanishingIdeal_eq_closure]
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      ⊢ Membership.mem (minimalPrimes R) (PrimeSpectrum.vanishingIdeal s) → Membersh …
    -/
    exact Set.mem_image_of_mem _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      ⊢ Membership.mem (irreducibleComponents (PrimeSpectrum R)) (closure s) → Membe …
    -/
  · rw [← vanishingIdeal_irreducibleComponents, ← vanishingIdeal_closure]
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      s : Set (PrimeSpectrum R)
      ⊢ Membership.mem (irreducibleComponents (PrimeSpectrum R)) (closure s) → Membe …
    -/
    exact Set.mem_image_of_mem _
    /-
      🎉 no goals
    -/


lemma zeroLocus_ideal_mem_irreducibleComponents {I : Ideal R} :
    zeroLocus I ∈ irreducibleComponents (PrimeSpectrum R) ↔ I.radical ∈ minimalPrimes R := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Iff (Membership.mem (irreducibleComponents (PrimeSpectrum R)) (PrimeSpectrum …
  -/
  rw [← vanishingIdeal_zeroLocus_eq_radical]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Iff (Membership.mem (irreducibleComponents (PrimeSpectrum R)) (PrimeSpectrum …
  -/
  conv_lhs => rw [← (isClosed_zeroLocus _).closure_eq]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Iff (Membership.mem (irreducibleComponents (PrimeSpectrum R)) (closure (Prim …
  -/
  exact vanishingIdeal_mem_minimalPrimes.symm
  /-
    🎉 no goals
  -/


/-- The closed point in the prime spectrum of a local ring. -/
def closedPoint : PrimeSpectrum R :=
  ⟨maximalIdeal R, (maximalIdeal.isMaximal R).isPrime⟩


theorem isLocalHom_iff_comap_closedPoint {S : Type v} [CommSemiring S] [IsLocalRing S]
    (f : R →+* S) : IsLocalHom f ↔ PrimeSpectrum.comap f (closedPoint S) = closedPoint R := by
  -- Porting note: inline `this` does **not** work
  /-
    R : Type u
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    ⊢ Iff (IsLocalHom f) (Eq ((PrimeSpectrum.comap f) (IsLocalRing.closedPoint S)) …
  -/
  have := (local_hom_TFAE f).out 0 4
  /-
    R : Type u
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    this : Iff (IsLocalHom f) (Eq (Ideal.comap f (IsLocalRing.maximalIdeal S)) (Is …
    ⊢ Iff (IsLocalHom f) (Eq ((PrimeSpectrum.comap f) (IsLocalRing.closedPoint S)) …
  -/
  rw [this, PrimeSpectrum.ext_iff]
  /-
    R : Type u
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    this : Iff (IsLocalHom f) (Eq (Ideal.comap f (IsLocalRing.maximalIdeal S)) (Is …
    ⊢ Iff (Eq (Ideal.comap f (IsLocalRing.maximalIdeal S)) (IsLocalRing.maximalIde …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_iff_comap_closedPoint := isLocalHom_iff_comap_closedPoint


@[simp]
theorem comap_closedPoint {S : Type v} [CommSemiring S] [IsLocalRing S] (f : R →+* S)
    [IsLocalHom f] : PrimeSpectrum.comap f (closedPoint S) = closedPoint R :=
  (isLocalHom_iff_comap_closedPoint f).mp inferInstance


theorem specializes_closedPoint (x : PrimeSpectrum R) : x ⤳ closedPoint R :=
  (PrimeSpectrum.le_iff_specializes _ _).mp (IsLocalRing.le_maximalIdeal x.2.1)


theorem closedPoint_mem_iff (U : TopologicalSpace.Opens <| PrimeSpectrum R) :
    closedPoint R ∈ U ↔ U = ⊤ := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : IsLocalRing R
    U : TopologicalSpace.Opens (PrimeSpectrum R)
    ⊢ Iff (Membership.mem U (IsLocalRing.closedPoint R)) (Eq U Top.top)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      U : TopologicalSpace.Opens (PrimeSpectrum R)
      ⊢ Membership.mem U (IsLocalRing.closedPoint R) → Eq U Top.top
    -/
  · rw [eq_top_iff]
    /-
      case mp
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      U : TopologicalSpace.Opens (PrimeSpectrum R)
      ⊢ Membership.mem U (IsLocalRing.closedPoint R) → LE.le Top.top U
    -/
    exact fun h x _ => (specializes_closedPoint x).mem_open U.2 h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      U : TopologicalSpace.Opens (PrimeSpectrum R)
      ⊢ Eq U Top.top → Membership.mem U (IsLocalRing.closedPoint R)
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : IsLocalRing R
      ⊢ Membership.mem Top.top (IsLocalRing.closedPoint R)
    -/
    trivial
    /-
      🎉 no goals
    -/


lemma closed_point_mem_iff {U : TopologicalSpace.Opens (PrimeSpectrum R)} :
    closedPoint R ∈ U ↔ U = ⊤ :=
  ⟨(eq_top_iff.mpr fun x _ ↦ (specializes_closedPoint x).mem_open U.2 ·), (· ▸ trivial)⟩


@[simp]
theorem PrimeSpectrum.comap_residue (T : Type u) [CommRing T] [IsLocalRing T]
    (x : PrimeSpectrum (ResidueField T)) : PrimeSpectrum.comap (residue T) x = closedPoint T := by
  /-
    T : Type u
    inst✝¹ : CommRing T
    inst✝ : IsLocalRing T
    x : PrimeSpectrum (IsLocalRing.ResidueField T)
    ⊢ Eq ((PrimeSpectrum.comap (IsLocalRing.residue T)) x) (IsLocalRing.closedPoin …
  -/
  rw [Subsingleton.elim x ⊥]
  /-
    T : Type u
    inst✝¹ : CommRing T
    inst✝ : IsLocalRing T
    x : PrimeSpectrum (IsLocalRing.ResidueField T)
    ⊢ Eq ((PrimeSpectrum.comap (IsLocalRing.residue T)) Bot.bot) (IsLocalRing.clos …
  -/
  ext1
  /-
    case asIdeal
    T : Type u
    inst✝¹ : CommRing T
    inst✝ : IsLocalRing T
    x : PrimeSpectrum (IsLocalRing.ResidueField T)
    ⊢ Eq ((PrimeSpectrum.comap (IsLocalRing.residue T)) Bot.bot).asIdeal (IsLocalR …
  -/
  exact Ideal.mk_ker
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-11")]
alias LocalRing.closedPoint := IsLocalRing.closedPoint


@[deprecated (since := "2024-11-11")]
alias LocalRing.isLocalHom_iff_comap_closedPoint := IsLocalRing.isLocalHom_iff_comap_closedPoint


@[deprecated (since := "2024-11-11")]
alias LocalRing.comap_closedPoint := IsLocalRing.comap_closedPoint


@[deprecated (since := "2024-11-11")]
alias LocalRing.specializes_closedPoint := IsLocalRing.specializes_closedPoint


@[deprecated (since := "2024-11-11")]
alias LocalRing.closedPoint_mem_iff := IsLocalRing.closedPoint_mem_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.closed_point_mem_iff := IsLocalRing.closed_point_mem_iff


@[deprecated (since := "2024-11-11")]
alias LocalRing.PrimeSpectrum.comap_residue := IsLocalRing.PrimeSpectrum.comap_residue


theorem PrimeSpectrum.topologicalKrullDim_eq_ringKrullDim [CommRing R] :
    topologicalKrullDim (PrimeSpectrum R) = ringKrullDim R :=
  Order.krullDim_orderDual.symm.trans <| Order.krullDim_eq_of_orderIso
  (PrimeSpectrum.pointsEquivIrreducibleCloseds R).symm


@[stacks 00EC]
lemma basicOpen_eq_zeroLocus_of_isIdempotentElem
    (e : R) (he : IsIdempotentElem e) :
    basicOpen e = zeroLocus {1 - e} := by
  /-
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    ⊢ Eq (↑(PrimeSpectrum.basicOpen e)) (PrimeSpectrum.zeroLocus (Singleton.single …
  -/
  ext p
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    p : PrimeSpectrum R
    ⊢ Iff (Membership.mem (↑(PrimeSpectrum.basicOpen e)) p) (Membership.mem (Prime …
  -/
  suffices e ∉ p.asIdeal ↔ 1 - e ∈ p.asIdeal by simpa
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    p : PrimeSpectrum R
    ⊢ Iff (Not (Membership.mem p.asIdeal e)) (Membership.mem p.asIdeal (HSub.hSub  …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      inst✝ : CommRing R
      e : R
      he : IsIdempotentElem e
      p : PrimeSpectrum R
      ⊢ Not (Membership.mem p.asIdeal e) → Membership.mem p.asIdeal (HSub.hSub 1 e)
    -/
  · refine (p.2.mem_or_mem_of_mul_eq_zero ?_).resolve_left
    /-
      case h.mp
      R : Type u
      inst✝ : CommRing R
      e : R
      he : IsIdempotentElem e
      p : PrimeSpectrum R
      ⊢ Eq (HMul.hMul e (HSub.hSub 1 e)) 0
    -/
    rw [mul_sub, mul_one, he.eq, sub_self]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      inst✝ : CommRing R
      e : R
      he : IsIdempotentElem e
      p : PrimeSpectrum R
      ⊢ Membership.mem p.asIdeal (HSub.hSub 1 e) → Not (Membership.mem p.asIdeal e)
    -/
  · refine fun h₁ h₂ ↦ p.2.1 ?_
    /-
      case h.mpr
      R : Type u
      inst✝ : CommRing R
      e : R
      he : IsIdempotentElem e
      p : PrimeSpectrum R
      h₁ : Membership.mem p.asIdeal (HSub.hSub 1 e)
      h₂ : Membership.mem p.asIdeal e
      ⊢ Eq p.asIdeal Top.top
    -/
    rw [Ideal.eq_top_iff_one, ← sub_add_cancel 1 e]
    /-
      case h.mpr
      R : Type u
      inst✝ : CommRing R
      e : R
      he : IsIdempotentElem e
      p : PrimeSpectrum R
      h₁ : Membership.mem p.asIdeal (HSub.hSub 1 e)
      h₂ : Membership.mem p.asIdeal e
      ⊢ Membership.mem p.asIdeal (HAdd.hAdd (HSub.hSub 1 e) e)
    -/
    exact add_mem h₁ h₂
    /-
      🎉 no goals
    -/


@[stacks 00EC]
lemma zeroLocus_eq_basicOpen_of_isIdempotentElem
    (e : R) (he : IsIdempotentElem e) :
    zeroLocus {e} = basicOpen (1 - e) := by
  /-
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    ⊢ Eq (PrimeSpectrum.zeroLocus (Singleton.singleton e)) ↑(PrimeSpectrum.basicOp …
  -/
  rw [basicOpen_eq_zeroLocus_of_isIdempotentElem _ he.one_sub, sub_sub_cancel]
  /-
    🎉 no goals
  -/


lemma isClopen_iff {s : Set (PrimeSpectrum R)} :
    IsClopen s ↔ ∃ e : R, IsIdempotentElem e ∧ s = basicOpen e := by
  /-
    R : Type u
    inst✝ : CommRing R
    s : Set (PrimeSpectrum R)
    ⊢ Iff (IsClopen s) (Exists fun e => And (IsIdempotentElem e) (Eq s ↑(PrimeSpec …
  -/
  refine ⟨exists_idempotent_basicOpen_eq_of_isClopen, ?_⟩
  /-
    R : Type u
    inst✝ : CommRing R
    s : Set (PrimeSpectrum R)
    ⊢ (Exists fun e => And (IsIdempotentElem e) (Eq s ↑(PrimeSpectrum.basicOpen e) …
  -/
  rintro ⟨e, he, rfl⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    ⊢ IsClopen ↑(PrimeSpectrum.basicOpen e)
  -/
  refine ⟨?_, (basicOpen e).2⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    ⊢ IsClosed ↑(PrimeSpectrum.basicOpen e)
  -/
  rw [PrimeSpectrum.basicOpen_eq_zeroLocus_of_isIdempotentElem e he]
  /-
    case intro.intro
    R : Type u
    inst✝ : CommRing R
    e : R
    he : IsIdempotentElem e
    ⊢ IsClosed (PrimeSpectrum.zeroLocus (Singleton.singleton (HSub.hSub 1 e)))
  -/
  exact isClosed_zeroLocus _
  /-
    🎉 no goals
  -/


lemma isClopen_iff_zeroLocus {s : Set (PrimeSpectrum R)} :
    IsClopen s ↔ ∃ e : R, IsIdempotentElem e ∧ s = zeroLocus {e} :=
  isClopen_iff.trans <| ⟨fun ⟨e, he, h⟩ ↦ ⟨1 - e, he.one_sub,
    h.trans (basicOpen_eq_zeroLocus_of_isIdempotentElem e he)⟩,
    fun ⟨e, he, h⟩ ↦ ⟨1 - e, he.one_sub, h.trans (zeroLocus_eq_basicOpen_of_isIdempotentElem e he)⟩⟩


/-- Clopen subsets in the prime spectrum of a commutative ring are in 1-1 correspondence
with idempotent elements in the ring. -/
@[stacks 00EE]
def isIdempotentElemEquivClopens :
    {e : R | IsIdempotentElem e} ≃ Clopens (PrimeSpectrum R) :=
  .ofBijective (fun e ↦ ⟨basicOpen e.1, isClopen_iff.mpr ⟨_, e.2, rfl⟩⟩)
    ⟨fun x y eq ↦ Subtype.ext (basicOpen_injOn_isIdempotentElem x.2 y.2 <|
      SetLike.ext' (congr_arg (·.1) eq)), fun s ↦
        have ⟨e, he, h⟩ := exists_idempotent_basicOpen_eq_of_isClopen s.2
        ⟨⟨e, he⟩, Clopens.ext h.symm⟩⟩


lemma basicOpen_isIdempotentElemEquivClopens_symm (s) :
    basicOpen (isIdempotentElemEquivClopens (R := R).symm s).1 = s.toOpens :=
  Opens.ext <| congr_arg (·.1) (isIdempotentElemEquivClopens.apply_symm_apply s)


lemma coe_isIdempotentElemEquivClopens_apply (e) :
    (isIdempotentElemEquivClopens e : Set (PrimeSpectrum R)) = basicOpen (e.1 : R) := rfl


lemma isIdempotentElemEquivClopens_apply_toOpens (e) :
    (isIdempotentElemEquivClopens e).toOpens = basicOpen (e.1 : R) := rfl


lemma isIdempotentElemEquivClopens_mul (e₁ e₂ : {e : R | IsIdempotentElem e}) :
    isIdempotentElemEquivClopens ⟨_, e₁.2.mul e₂.2⟩ =
      isIdempotentElemEquivClopens e₁ ⊓ isIdempotentElemEquivClopens e₂ :=
                    /-
                      R : Type u
                      inst✝ : CommRing R
                      e₁ e₂ : ↑(setOf fun e => IsIdempotentElem e)
                      ⊢ Eq ↑(PrimeSpectrum.isIdempotentElemEquivClopens ⟨HMul.hMul ↑e₁ ↑e₂, ⋯⟩) ↑(Mi …
                    -/
  Clopens.ext <| by simp_rw [coe_isIdempotentElemEquivClopens_apply, basicOpen_mul]; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma isIdempotentElemEquivClopens_one_sub (e : {e : R | IsIdempotentElem e}) :
    isIdempotentElemEquivClopens ⟨_, e.2.one_sub⟩ = (isIdempotentElemEquivClopens e)ᶜ :=
  SetLike.ext' <| by
    /-
      R : Type u
      inst✝ : CommRing R
      e : ↑(setOf fun e => IsIdempotentElem e)
      ⊢ Eq ↑(PrimeSpectrum.isIdempotentElemEquivClopens ⟨HSub.hSub 1 ↑e, ⋯⟩) ↑(HasCo …
    -/
    simp_rw [Clopens.coe_compl, coe_isIdempotentElemEquivClopens_apply]
    /-
      R : Type u
      inst✝ : CommRing R
      e : ↑(setOf fun e => IsIdempotentElem e)
      ⊢ Eq (↑(PrimeSpectrum.basicOpen (HSub.hSub 1 ↑e))) (HasCompl.compl ↑(PrimeSpec …
    -/
    rw [basicOpen_eq_zeroLocus_compl, basicOpen_eq_zeroLocus_of_isIdempotentElem _ e.2]
    /-
      🎉 no goals
    -/


lemma isIdempotentElemEquivClopens_symm_inf (s₁ s₂) :
    letI e := isIdempotentElemEquivClopens (R := R).symm
    e (s₁ ⊓ s₂) = ⟨_, (e s₁).2.mul (e s₂).2⟩ :=
  isIdempotentElemEquivClopens.symm_apply_eq.mpr <| by
    /-
      R : Type u
      inst✝ : CommRing R
      s₁ s₂ : TopologicalSpace.Clopens (PrimeSpectrum R)
      ⊢ Eq (Min.min s₁ s₂) (PrimeSpectrum.isIdempotentElemEquivClopens ⟨HMul.hMul ↑( …
    -/
    simp_rw [isIdempotentElemEquivClopens_mul, Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


lemma isIdempotentElemEquivClopens_symm_compl (s : Clopens (PrimeSpectrum R)) :
    isIdempotentElemEquivClopens.symm sᶜ = ⟨_, (isIdempotentElemEquivClopens.symm s).2.one_sub⟩ :=
  isIdempotentElemEquivClopens.symm_apply_eq.mpr <| by
    /-
      R : Type u
      inst✝ : CommRing R
      s : TopologicalSpace.Clopens (PrimeSpectrum R)
      ⊢ Eq (HasCompl.compl s) (PrimeSpectrum.isIdempotentElemEquivClopens ⟨HSub.hSub …
    -/
    rw [isIdempotentElemEquivClopens_one_sub, Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


lemma isIdempotentElemEquivClopens_symm_top :
    isIdempotentElemEquivClopens.symm ⊤ = ⟨(1 : R), .one⟩ :=
  isIdempotentElemEquivClopens.symm_apply_eq.mpr <| Clopens.ext <| by
    /-
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq ↑Top.top ↑(PrimeSpectrum.isIdempotentElemEquivClopens ⟨1, ⋯⟩)
    -/
    rw [coe_isIdempotentElemEquivClopens_apply, basicOpen_one]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma isIdempotentElemEquivClopens_symm_bot :
    isIdempotentElemEquivClopens.symm ⊥ = ⟨(0 : R), .zero⟩ :=
  isIdempotentElemEquivClopens.symm_apply_eq.mpr <| Clopens.ext <| by
    /-
      R : Type u
      inst✝ : CommRing R
      ⊢ Eq ↑Bot.bot ↑(PrimeSpectrum.isIdempotentElemEquivClopens ⟨0, ⋯⟩)
    -/
    rw [coe_isIdempotentElemEquivClopens_apply, basicOpen_zero]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma isIdempotentElemEquivClopens_symm_sup (s₁ s₂ : Clopens (PrimeSpectrum R)) :
    letI e := isIdempotentElemEquivClopens (R := R).symm
    e (s₁ ⊔ s₂) = ⟨_, (e s₁).2.add_sub_mul (e s₂).2⟩ := Subtype.ext <| by
  /-
    R : Type u
    inst✝ : CommRing R
    s₁ s₂ : TopologicalSpace.Clopens (PrimeSpectrum R)
    ⊢ Eq ↑(PrimeSpectrum.isIdempotentElemEquivClopens.symm (Max.max s₁ s₂)) ↑⟨HSub …
  -/
  rw [← compl_compl (_ ⊔ _), compl_sup, isIdempotentElemEquivClopens_symm_compl]
  /-
    R : Type u
    inst✝ : CommRing R
    s₁ s₂ : TopologicalSpace.Clopens (PrimeSpectrum R)
    ⊢ Eq ↑⟨HSub.hSub 1 ↑(PrimeSpectrum.isIdempotentElemEquivClopens.symm (Min.min  …
  -/
  simp_rw [isIdempotentElemEquivClopens_symm_inf, isIdempotentElemEquivClopens_symm_compl]
  /-
    R : Type u
    inst✝ : CommRing R
    s₁ s₂ : TopologicalSpace.Clopens (PrimeSpectrum R)
    ⊢ Eq (HSub.hSub 1 (HMul.hMul (HSub.hSub 1 ↑(PrimeSpectrum.isIdempotentElemEqui …
  -/
  ring
  /-
    🎉 no goals
  -/


