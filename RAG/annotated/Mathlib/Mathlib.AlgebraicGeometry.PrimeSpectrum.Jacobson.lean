lemma exists_isClosed_singleton_of_isJacobsonRing [IsJacobsonRing R]
    (s : (Set (PrimeSpectrum R))) (hs : IsOpen s) (hs' : s.Nonempty) :
    ∃ x ∈ s, IsClosed {x} := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    ⊢ Exists fun x => And (Membership.mem s x) (IsClosed (Singleton.singleton x))
  -/
  simp_rw [isClosed_singleton_iff_isMaximal]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    ⊢ Exists fun x => And (Membership.mem s x) x.asIdeal.IsMaximal
  -/
  obtain ⟨I, hI'⟩ := (isClosed_iff_zeroLocus_ideal _).mp hs.isClosed_compl
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    ⊢ Exists fun x => And (Membership.mem s x) x.asIdeal.IsMaximal
  -/
  simp_rw [← @Set.not_mem_compl_iff _ s, hI', mem_zeroLocus]
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    ⊢ Exists fun x => And (Not (HasSubset.Subset ↑I ↑x.asIdeal)) x.asIdeal.IsMaximal
  -/
  have := hs'.ne_empty
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    this : Ne s EmptyCollection.emptyCollection
    ⊢ Exists fun x => And (Not (HasSubset.Subset ↑I ↑x.asIdeal)) x.asIdeal.IsMaximal
  -/
  contrapose! this
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    this : ∀ (x : PrimeSpectrum R), Not (HasSubset.Subset ↑I ↑x.asIdeal) → Not x.a …
    ⊢ Eq s EmptyCollection.emptyCollection
  -/
  simp_rw [not_imp_not] at this
  rw [← Set.compl_univ, eq_compl_comm, hI', eq_comm, ← zeroLocus_bot,
    zeroLocus_eq_iff, Ideal.radical_eq_jacobson, Ideal.radical_eq_jacobson]
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    this : ∀ (x : PrimeSpectrum R), x.asIdeal.IsMaximal → HasSubset.Subset ↑I ↑x.a …
    ⊢ Eq I.jacobson Bot.bot.jacobson
  -/
  refine le_antisymm (le_sInf ?_) (Ideal.jacobson_mono bot_le)
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    this : ∀ (x : PrimeSpectrum R), x.asIdeal.IsMaximal → HasSubset.Subset ↑I ↑x.a …
    ⊢ ∀ (b : Ideal R), Membership.mem (setOf fun J => And (LE.le Bot.bot J) J.IsMa …
  -/
  rintro x ⟨-, hx⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    s : Set (PrimeSpectrum R)
    hs : IsOpen s
    hs' : s.Nonempty
    I : Ideal R
    hI' : Eq (HasCompl.compl s) (PrimeSpectrum.zeroLocus ↑I)
    this : ∀ (x : PrimeSpectrum R), x.asIdeal.IsMaximal → HasSubset.Subset ↑I ↑x.a …
    x : Ideal R
    hx : x.IsMaximal
    ⊢ LE.le I.jacobson x
  -/
  exact sInf_le ⟨this ⟨x, hx.isPrime⟩ hx, hx⟩
  /-
    🎉 no goals
  -/


instance [IsJacobsonRing R] : JacobsonSpace (PrimeSpectrum R) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    ⊢ JacobsonSpace (PrimeSpectrum R)
  -/
  rw [jacobsonSpace_iff_locallyClosed]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    ⊢ ∀ (Z : Set (PrimeSpectrum R)), Z.Nonempty → IsLocallyClosed Z → (Inter.inter …
  -/
  rintro S hS ⟨U, Z, hU, hZ, rfl⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    U Z : Set (PrimeSpectrum R)
    hU : IsOpen U
    hZ : IsClosed Z
    hS : (Inter.inter U Z).Nonempty
    ⊢ (Inter.inter (Inter.inter U Z) (closedPoints (PrimeSpectrum R))).Nonempty
  -/
  simp only [← isClosed_compl_iff, isClosed_iff_zeroLocus_ideal, @compl_eq_comm _ U] at hU hZ
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    U Z : Set (PrimeSpectrum R)
    hS : (Inter.inter U Z).Nonempty
    hU : Exists fun I => Eq (HasCompl.compl (PrimeSpectrum.zeroLocus ↑I)) U
    hZ : Exists fun I => Eq Z (PrimeSpectrum.zeroLocus ↑I)
    ⊢ (Inter.inter (Inter.inter U Z) (closedPoints (PrimeSpectrum R))).Nonempty
  -/
  obtain ⟨⟨I, rfl⟩, ⟨J, rfl⟩⟩ := And.intro hU hZ
  simp only [Set.nonempty_iff_ne_empty, ne_eq, Set.inter_assoc,
    ← Set.disjoint_iff_inter_eq_empty, Set.disjoint_compl_left_iff_subset,
    zeroLocus_subset_zeroLocus_iff, Ideal.radical_eq_jacobson, Ideal.jacobson, le_sInf_iff] at hS ⊢
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    I : Ideal R
    hU : Exists fun I_1 => Eq (HasCompl.compl (PrimeSpectrum.zeroLocus ↑I_1)) (Has …
    J : Ideal R
    hZ : Exists fun I => Eq (PrimeSpectrum.zeroLocus ↑J) (PrimeSpectrum.zeroLocus  …
    hS : Not (∀ (b : Ideal R), Membership.mem (setOf fun J_1 => And (LE.le J J_1)  …
    ⊢ Not (HasSubset.Subset (Inter.inter (PrimeSpectrum.zeroLocus ↑J) (closedPoint …
  -/
  contrapose! hS
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    I : Ideal R
    hU : Exists fun I_1 => Eq (HasCompl.compl (PrimeSpectrum.zeroLocus ↑I_1)) (Has …
    J : Ideal R
    hZ : Exists fun I => Eq (PrimeSpectrum.zeroLocus ↑J) (PrimeSpectrum.zeroLocus  …
    hS : HasSubset.Subset (Inter.inter (PrimeSpectrum.zeroLocus ↑J) (closedPoints  …
    ⊢ ∀ (b : Ideal R), Membership.mem (setOf fun J_1 => And (LE.le J J_1) J_1.IsMa …
  -/
  rintro x ⟨hJx, hx⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsJacobsonRing R
    I : Ideal R
    hU : Exists fun I_1 => Eq (HasCompl.compl (PrimeSpectrum.zeroLocus ↑I_1)) (Has …
    J : Ideal R
    hZ : Exists fun I => Eq (PrimeSpectrum.zeroLocus ↑J) (PrimeSpectrum.zeroLocus  …
    hS : HasSubset.Subset (Inter.inter (PrimeSpectrum.zeroLocus ↑J) (closedPoints  …
    x : Ideal R
    hJx : LE.le J x
    hx : x.IsMaximal
    ⊢ LE.le I x
  -/
  exact @hS ⟨x, hx.isPrime⟩ ⟨hJx, (isClosed_singleton_iff_isMaximal _).mpr hx⟩
  /-
    🎉 no goals
  -/


lemma isJacobsonRing_iff_jacobsonSpace :
    IsJacobsonRing R ↔ JacobsonSpace (PrimeSpectrum R) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Iff (IsJacobsonRing R) (JacobsonSpace (PrimeSpectrum R))
  -/
  refine ⟨fun _ ↦ inferInstance, fun H ↦ ⟨fun I hI ↦ le_antisymm ?_ Ideal.le_jacobson⟩⟩
  /-
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    ⊢ LE.le I.jacobson I
  -/
  rw [← I.isRadical_jacobson.radical]
  /-
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    ⊢ LE.le I.jacobson.radical I
  -/
  conv_rhs => rw [← hI.radical]
  /-
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    ⊢ LE.le I.jacobson.radical I.radical
  -/
  simp_rw [← vanishingIdeal_zeroLocus_eq_radical]
  /-
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    ⊢ LE.le (PrimeSpectrum.vanishingIdeal (PrimeSpectrum.zeroLocus ↑I.jacobson)) ( …
  -/
  apply vanishingIdeal_anti_mono
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    ⊢ HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (PrimeSpectrum.zeroLocus ↑I.ja …
  -/
  rw [← H.1 (isClosed_zeroLocus I), (isClosed_zeroLocus _).closure_subset_iff]
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    ⊢ HasSubset.Subset (Inter.inter (PrimeSpectrum.zeroLocus ↑I) (closedPoints (Pr …
  -/
  rintro x ⟨hx : I ≤ x.asIdeal, hx'⟩
  /-
    case h.intro
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    x : PrimeSpectrum R
    hx : LE.le I x.asIdeal
    hx' : Membership.mem (closedPoints (PrimeSpectrum R)) x
    ⊢ Membership.mem (PrimeSpectrum.zeroLocus ↑I.jacobson) x
  -/
  show jacobson I ≤ x.asIdeal
  /-
    case h.intro
    R : Type u_1
    inst✝ : CommRing R
    H : JacobsonSpace (PrimeSpectrum R)
    I : Ideal R
    hI : I.IsRadical
    x : PrimeSpectrum R
    hx : LE.le I x.asIdeal
    hx' : Membership.mem (closedPoints (PrimeSpectrum R)) x
    ⊢ LE.le I.jacobson x.asIdeal
  -/
  exact sInf_le ⟨hx, (isClosed_singleton_iff_isMaximal _).mp hx'⟩
  /-
    🎉 no goals
  -/


/--
If `R` is both noetherian and jacobson, then the following are equivalent for `x : Spec R`:
1. `{x}` is open (i.e. `x` is an isolated point)
2. `{x}` is clopen
3. `{x}` is both closed and stable under generalization
  (i.e. `x` is both a minimal prime and a maximal ideal)
-/
lemma isOpen_singleton_tfae_of_isNoetherian_of_isJacobsonRing
    [IsNoetherianRing R] [IsJacobsonRing R] (x : PrimeSpectrum R) :
    List.TFAE [IsOpen {x}, IsClopen {x}, IsClosed {x} ∧ StableUnderGeneralization {x}] := by
  tfae_have 1 → 2
  | h => by
    obtain ⟨y, rfl : y = x, h'⟩ := exists_isClosed_singleton_of_isJacobsonRing _ h
      ⟨x, Set.mem_singleton x⟩
    exact ⟨h', h⟩
  tfae_have 2 → 3
  | h => ⟨h.isClosed, h.isOpen.stableUnderGeneralization⟩
  tfae_have 3 → 1
  | ⟨h₁, h₂⟩ => by
    rw [isClosed_singleton_iff_isMaximal, ← isMax_iff] at h₁
    suffices {x} = (⋃ p ∈ { p : PrimeSpectrum R | IsMin p ∧ p ≠ x }, closure {p})ᶜ by
      rw [this, isOpen_compl_iff]
      refine Set.Finite.isClosed_biUnion ?_ (fun _ _ ↦ isClosed_closure)
      exact (finite_setOf_isMin R).subset fun x h ↦ h.1
    ext p
    simp only [Set.mem_singleton_iff, ne_eq, Set.mem_setOf_eq, Set.compl_iUnion, Set.mem_iInter,
      Set.mem_compl_iff, and_imp, ← specializes_iff_mem_closure, ← le_iff_specializes,
      not_imp_not]
    constructor
    · rintro rfl _ _
      rw [stableUnderGeneralization_singleton, ← isMin_iff] at h₂
      exact h₂.eq_of_le
    · intros hp
      apply h₁.eq_of_ge
      obtain ⟨q, hq, hq'⟩ := Ideal.exists_minimalPrimes_le (J := p.asIdeal) bot_le
      exact (hp ⟨q, hq.1.1⟩ (isMin_iff.mpr hq) hq').ge.trans hq'
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsJacobsonRing R
    x : PrimeSpectrum R
    tfae_1_to_2 : IsOpen (Singleton.singleton x) → IsClopen (Singleton.singleton x)
    tfae_2_to_3 : IsClopen (Singleton.singleton x) → And (IsClosed (Singleton.sing …
    tfae_3_to_1 : And (IsClosed (Singleton.singleton x)) (StableUnderGeneralizatio …
    ⊢ (List.cons (IsOpen (Singleton.singleton x)) (List.cons (IsClopen (Singleton. …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


