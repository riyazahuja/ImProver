/-- The prime spectrum of a commutative (semi)ring `R` is the type of all prime ideals of `R`.

It is naturally endowed with a topology (the Zariski topology),
and a sheaf of commutative rings (see `AlgebraicGeometry.StructureSheaf`).
It is a fundamental building block in algebraic geometry. -/
@[ext]
structure PrimeSpectrum [CommSemiring R] where
  asIdeal : Ideal R
  isPrime : asIdeal.IsPrime


@[deprecated (since := "2024-06-22")] alias PrimeSpectrum.IsPrime := PrimeSpectrum.isPrime


instance [Nontrivial R] : Nonempty <| PrimeSpectrum R :=
  let ⟨I, hI⟩ := Ideal.exists_maximal R
  ⟨⟨I, hI.isPrime⟩⟩


/-- The prime spectrum of the zero ring is empty. -/
instance [Subsingleton R] : IsEmpty (PrimeSpectrum R) :=
  ⟨fun x ↦ x.isPrime.ne_top <| SetLike.ext' <| Subsingleton.eq_univ_of_nonempty x.asIdeal.nonempty⟩


/-- The prime spectrum is in bijection with the set of prime ideals. -/
@[simps]
def equivSubtype : PrimeSpectrum R ≃ {I : Ideal R // I.IsPrime} where
  toFun I := ⟨I.asIdeal, I.2⟩
  invFun I := ⟨I, I.2⟩
  left_inv _ := rfl
  right_inv _ := rfl


/-- The map from the direct sum of prime spectra to the prime spectrum of a direct product. -/
@[simp]
def primeSpectrumProdOfSum : PrimeSpectrum R ⊕ PrimeSpectrum S → PrimeSpectrum (R × S)
  | Sum.inl ⟨I, _⟩ => ⟨Ideal.prod I ⊤, Ideal.isPrime_ideal_prod_top⟩
  | Sum.inr ⟨J, _⟩ => ⟨Ideal.prod ⊤ J, Ideal.isPrime_ideal_prod_top'⟩


/-- The prime spectrum of `R × S` is in bijection with the disjoint unions of the prime spectrum of
`R` and the prime spectrum of `S`. -/
noncomputable def primeSpectrumProd :
    PrimeSpectrum (R × S) ≃ PrimeSpectrum R ⊕ PrimeSpectrum S :=
  Equiv.symm <|
    Equiv.ofBijective (primeSpectrumProdOfSum R S) (by
        /-
          R : Type u
          S : Type v
          inst✝¹ : CommSemiring R
          inst✝ : CommSemiring S
          ⊢ Function.Bijective (PrimeSpectrum.primeSpectrumProdOfSum R S)
        -/
        constructor
          /-
            case left
            R : Type u
            S : Type v
            inst✝¹ : CommSemiring R
            inst✝ : CommSemiring S
            ⊢ Function.Injective (PrimeSpectrum.primeSpectrumProdOfSum R S)
          -/
        · rintro (⟨I, hI⟩ | ⟨J, hJ⟩) (⟨I', hI'⟩ | ⟨J', hJ'⟩) h <;>
          /-
            case left.inl.mk.inl.mk
            R : Type u
            S : Type v
            inst✝¹ : CommSemiring R
            inst✝ : CommSemiring S
            I : Ideal R
            hI : I.IsPrime
            I' : Ideal R
            hI' : I'.IsPrime
            h : Eq (PrimeSpectrum.primeSpectrumProdOfSum R S (Sum.inl { asIdeal := I, isPr …
            ⊢ Eq (Sum.inl { asIdeal := I, isPrime := hI }) (Sum.inl { asIdeal := I', isPri …
          -/
          simp only [mk.injEq, Ideal.prod.ext_iff, primeSpectrumProdOfSum] at h
            /-
              case left.inl.mk.inl.mk
              R : Type u
              S : Type v
              inst✝¹ : CommSemiring R
              inst✝ : CommSemiring S
              I : Ideal R
              hI : I.IsPrime
              I' : Ideal R
              hI' : I'.IsPrime
              h : And (Eq I I') True
              ⊢ Eq (Sum.inl { asIdeal := I, isPrime := hI }) (Sum.inl { asIdeal := I', isPri …
            -/
          · simp only [h]
            /-
              🎉 no goals
            -/
            /-
              case left.inl.mk.inr.mk
              R : Type u
              S : Type v
              inst✝¹ : CommSemiring R
              inst✝ : CommSemiring S
              I : Ideal R
              hI : I.IsPrime
              J' : Ideal S
              hJ' : J'.IsPrime
              h : And (Eq I Top.top) (Eq Top.top J')
              ⊢ Eq (Sum.inl { asIdeal := I, isPrime := hI }) (Sum.inr { asIdeal := J', isPri …
            -/
          · exact False.elim (hI.ne_top h.left)
            /-
              🎉 no goals
            -/
            /-
              case left.inr.mk.inl.mk
              R : Type u
              S : Type v
              inst✝¹ : CommSemiring R
              inst✝ : CommSemiring S
              J : Ideal S
              hJ : J.IsPrime
              I' : Ideal R
              hI' : I'.IsPrime
              h : And (Eq Top.top I') (Eq J Top.top)
              ⊢ Eq (Sum.inr { asIdeal := J, isPrime := hJ }) (Sum.inl { asIdeal := I', isPri …
            -/
          · exact False.elim (hJ.ne_top h.right)
            /-
              🎉 no goals
            -/
            /-
              case left.inr.mk.inr.mk
              R : Type u
              S : Type v
              inst✝¹ : CommSemiring R
              inst✝ : CommSemiring S
              J : Ideal S
              hJ : J.IsPrime
              J' : Ideal S
              hJ' : J'.IsPrime
              h : And True (Eq J J')
              ⊢ Eq (Sum.inr { asIdeal := J, isPrime := hJ }) (Sum.inr { asIdeal := J', isPri …
            -/
          · simp only [h]
            /-
              🎉 no goals
            -/
          /-
            case right
            R : Type u
            S : Type v
            inst✝¹ : CommSemiring R
            inst✝ : CommSemiring S
            ⊢ Function.Surjective (PrimeSpectrum.primeSpectrumProdOfSum R S)
          -/
        · rintro ⟨I, hI⟩
          /-
            case right.mk
            R : Type u
            S : Type v
            inst✝¹ : CommSemiring R
            inst✝ : CommSemiring S
            I : Ideal (Prod R S)
            hI : I.IsPrime
            ⊢ Exists fun a => Eq (PrimeSpectrum.primeSpectrumProdOfSum R S a) { asIdeal := …
          -/
          rcases (Ideal.ideal_prod_prime I).mp hI with (⟨p, ⟨hp, rfl⟩⟩ | ⟨p, ⟨hp, rfl⟩⟩)
            /-
              case right.mk.inl.intro.intro
              R : Type u
              S : Type v
              inst✝¹ : CommSemiring R
              inst✝ : CommSemiring S
              p : Ideal R
              hp : p.IsPrime
              hI : (p.prod Top.top).IsPrime
              ⊢ Exists fun a => Eq (PrimeSpectrum.primeSpectrumProdOfSum R S a) { asIdeal := …
            -/
          · exact ⟨Sum.inl ⟨p, hp⟩, rfl⟩
            /-
              🎉 no goals
            -/
            /-
              case right.mk.inr.intro.intro
              R : Type u
              S : Type v
              inst✝¹ : CommSemiring R
              inst✝ : CommSemiring S
              p : Ideal S
              hp : p.IsPrime
              hI : (Top.top.prod p).IsPrime
              ⊢ Exists fun a => Eq (PrimeSpectrum.primeSpectrumProdOfSum R S a) { asIdeal := …
            -/
          · exact ⟨Sum.inr ⟨p, hp⟩, rfl⟩)
            /-
              🎉 no goals
            -/


@[simp]
theorem primeSpectrumProd_symm_inl_asIdeal (x : PrimeSpectrum R) :
    ((primeSpectrumProd R S).symm <| Sum.inl x).asIdeal = Ideal.prod x.asIdeal ⊤ := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    x : PrimeSpectrum R
    ⊢ Eq ((PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inl x)).asIdeal (x.asIde …
  -/
  cases x
  /-
    case mk
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    asIdeal✝ : Ideal R
    isPrime✝ : asIdeal✝.IsPrime
    ⊢ Eq ((PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inl { asIdeal := asIdeal …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem primeSpectrumProd_symm_inr_asIdeal (x : PrimeSpectrum S) :
    ((primeSpectrumProd R S).symm <| Sum.inr x).asIdeal = Ideal.prod ⊤ x.asIdeal := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    x : PrimeSpectrum S
    ⊢ Eq ((PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inr x)).asIdeal (Top.top …
  -/
  cases x
  /-
    case mk
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    asIdeal✝ : Ideal S
    isPrime✝ : asIdeal✝.IsPrime
    ⊢ Eq ((PrimeSpectrum.primeSpectrumProd R S).symm (Sum.inr { asIdeal := asIdeal …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The zero locus of a set `s` of elements of a commutative (semi)ring `R` is the set of all
prime ideals of the ring that contain the set `s`.

An element `f` of `R` can be thought of as a dependent function on the prime spectrum of `R`.
At a point `x` (a prime ideal) the function (i.e., element) `f` takes values in the quotient ring
`R` modulo the prime ideal `x`. In this manner, `zeroLocus s` is exactly the subset of
`PrimeSpectrum R` where all "functions" in `s` vanish simultaneously.
-/
def zeroLocus (s : Set R) : Set (PrimeSpectrum R) :=
  { x | s ⊆ x.asIdeal }


@[simp]
theorem mem_zeroLocus (x : PrimeSpectrum R) (s : Set R) : x ∈ zeroLocus s ↔ s ⊆ x.asIdeal :=
  Iff.rfl


@[simp]
theorem zeroLocus_span (s : Set R) : zeroLocus (Ideal.span s : Set R) = zeroLocus s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Eq (PrimeSpectrum.zeroLocus ↑(Ideal.span s)) (PrimeSpectrum.zeroLocus s)
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    x : PrimeSpectrum R
    ⊢ Iff (Membership.mem (PrimeSpectrum.zeroLocus ↑(Ideal.span s)) x) (Membership …
  -/
  exact (Submodule.gi R R).gc s x.asIdeal
  /-
    🎉 no goals
  -/


/-- The vanishing ideal of a set `t` of points of the prime spectrum of a commutative ring `R` is
the intersection of all the prime ideals in the set `t`.

An element `f` of `R` can be thought of as a dependent function on the prime spectrum of `R`.
At a point `x` (a prime ideal) the function (i.e., element) `f` takes values in the quotient ring
`R` modulo the prime ideal `x`. In this manner, `vanishingIdeal t` is exactly the ideal of `R`
consisting of all "functions" that vanish on all of `t`.
-/
def vanishingIdeal (t : Set (PrimeSpectrum R)) : Ideal R :=
  ⨅ x ∈ t, x.asIdeal


theorem coe_vanishingIdeal (t : Set (PrimeSpectrum R)) :
    (vanishingIdeal t : Set R) = { f : R | ∀ x ∈ t, f ∈ x.asIdeal } := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    ⊢ Eq (↑(PrimeSpectrum.vanishingIdeal t)) (setOf fun f => ∀ (x : PrimeSpectrum  …
  -/
  ext f
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    f : R
    ⊢ Iff (Membership.mem (↑(PrimeSpectrum.vanishingIdeal t)) f) (Membership.mem ( …
  -/
  rw [vanishingIdeal, SetLike.mem_coe, Submodule.mem_iInf]
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    f : R
    ⊢ Iff (∀ (i : PrimeSpectrum R), Membership.mem (iInf fun h => i.asIdeal) f) (M …
  -/
  apply forall_congr'; intro x
  /-
    case h.h
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    f : R
    x : PrimeSpectrum R
    ⊢ Iff (Membership.mem (iInf fun h => x.asIdeal) f) (Membership.mem t x → Membe …
  -/
  rw [Submodule.mem_iInf]
  /-
    🎉 no goals
  -/


theorem mem_vanishingIdeal (t : Set (PrimeSpectrum R)) (f : R) :
    f ∈ vanishingIdeal t ↔ ∀ x ∈ t, f ∈ x.asIdeal := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Set (PrimeSpectrum R)
    f : R
    ⊢ Iff (Membership.mem (PrimeSpectrum.vanishingIdeal t) f) (∀ (x : PrimeSpectru …
  -/
  rw [← SetLike.mem_coe, coe_vanishingIdeal, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem vanishingIdeal_singleton (x : PrimeSpectrum R) :
                                                                   /-
                                                                     R : Type u
                                                                     inst✝ : CommSemiring R
                                                                     x : PrimeSpectrum R
                                                                     ⊢ Eq (PrimeSpectrum.vanishingIdeal (Singleton.singleton x)) x.asIdeal
                                                                   -/
    vanishingIdeal ({x} : Set (PrimeSpectrum R)) = x.asIdeal := by simp [vanishingIdeal]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem subset_zeroLocus_iff_le_vanishingIdeal (t : Set (PrimeSpectrum R)) (I : Ideal R) :
    t ⊆ zeroLocus I ↔ I ≤ vanishingIdeal t :=
  ⟨fun h _ k => (mem_vanishingIdeal _ _).mpr fun _ j => (mem_zeroLocus _ _).mpr (h j) k, fun h =>
    fun x j => (mem_zeroLocus _ _).mpr (le_trans h fun _ h => ((mem_vanishingIdeal _ _).mp h) x j)⟩


/-- `zeroLocus` and `vanishingIdeal` form a galois connection. -/
theorem gc :
    @GaloisConnection (Ideal R) (Set (PrimeSpectrum R))ᵒᵈ _ _ (fun I => zeroLocus I) fun t =>
      vanishingIdeal t :=
  fun I t => subset_zeroLocus_iff_le_vanishingIdeal t I


/-- `zeroLocus` and `vanishingIdeal` form a galois connection. -/
theorem gc_set :
    @GaloisConnection (Set R) (Set (PrimeSpectrum R))ᵒᵈ _ _ (fun s => zeroLocus s) fun t =>
      vanishingIdeal t := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ GaloisConnection (fun s => PrimeSpectrum.zeroLocus s) fun t => ↑(PrimeSpectr …
  -/
  have ideal_gc : GaloisConnection Ideal.span _ := (Submodule.gi R R).gc
  /-
    R : Type u
    inst✝ : CommSemiring R
    ideal_gc : GaloisConnection Ideal.span SetLike.coe
    ⊢ GaloisConnection (fun s => PrimeSpectrum.zeroLocus s) fun t => ↑(PrimeSpectr …
  -/
  simpa [zeroLocus_span, Function.comp_def] using ideal_gc.compose (gc R)
  /-
    🎉 no goals
  -/


theorem subset_zeroLocus_iff_subset_vanishingIdeal (t : Set (PrimeSpectrum R)) (s : Set R) :
    t ⊆ zeroLocus s ↔ s ⊆ vanishingIdeal t :=
  (gc_set R) s t


theorem subset_vanishingIdeal_zeroLocus (s : Set R) : s ⊆ vanishingIdeal (zeroLocus s) :=
  (gc_set R).le_u_l s


theorem le_vanishingIdeal_zeroLocus (I : Ideal R) : I ≤ vanishingIdeal (zeroLocus I) :=
  (gc R).le_u_l I


@[simp]
theorem vanishingIdeal_zeroLocus_eq_radical (I : Ideal R) :
    vanishingIdeal (zeroLocus (I : Set R)) = I.radical :=
  Ideal.ext fun f => by
    /-
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : R
      ⊢ Iff (Membership.mem (PrimeSpectrum.vanishingIdeal (PrimeSpectrum.zeroLocus ↑ …
    -/
    rw [mem_vanishingIdeal, Ideal.radical_eq_sInf, Submodule.mem_sInf]
    /-
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : R
      ⊢ Iff (∀ (x : PrimeSpectrum R), Membership.mem (PrimeSpectrum.zeroLocus ↑I) x  …
    -/
    exact ⟨fun h x hx => h ⟨x, hx.2⟩ hx.1, fun h x hx => h x.1 ⟨hx, x.2⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem zeroLocus_radical (I : Ideal R) : zeroLocus (I.radical : Set R) = zeroLocus I :=
  vanishingIdeal_zeroLocus_eq_radical I ▸ (gc R).l_u_l_eq_l I


theorem subset_zeroLocus_vanishingIdeal (t : Set (PrimeSpectrum R)) :
    t ⊆ zeroLocus (vanishingIdeal t) :=
  (gc R).l_u_le t


theorem zeroLocus_anti_mono {s t : Set R} (h : s ⊆ t) : zeroLocus t ⊆ zeroLocus s :=
  (gc_set R).monotone_l h


theorem zeroLocus_anti_mono_ideal {s t : Ideal R} (h : s ≤ t) :
    zeroLocus (t : Set R) ⊆ zeroLocus (s : Set R) :=
  (gc R).monotone_l h


theorem vanishingIdeal_anti_mono {s t : Set (PrimeSpectrum R)} (h : s ⊆ t) :
    vanishingIdeal t ≤ vanishingIdeal s :=
  (gc R).monotone_u h


theorem zeroLocus_subset_zeroLocus_iff (I J : Ideal R) :
    zeroLocus (I : Set R) ⊆ zeroLocus (J : Set R) ↔ J ≤ I.radical := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I J : Ideal R
    ⊢ Iff (HasSubset.Subset (PrimeSpectrum.zeroLocus ↑I) (PrimeSpectrum.zeroLocus  …
  -/
  rw [subset_zeroLocus_iff_le_vanishingIdeal, vanishingIdeal_zeroLocus_eq_radical]
  /-
    🎉 no goals
  -/


theorem zeroLocus_subset_zeroLocus_singleton_iff (f g : R) :
    zeroLocus ({f} : Set R) ⊆ zeroLocus {g} ↔ g ∈ (Ideal.span ({f} : Set R)).radical := by
  rw [← zeroLocus_span {f}, ← zeroLocus_span {g}, zeroLocus_subset_zeroLocus_iff, Ideal.span_le,
    Set.singleton_subset_iff, SetLike.mem_coe]


theorem zeroLocus_bot : zeroLocus ((⊥ : Ideal R) : Set R) = Set.univ :=
  (gc R).l_bot


@[simp]
theorem zeroLocus_singleton_zero : zeroLocus ({0} : Set R) = Set.univ :=
  zeroLocus_bot


@[simp]
theorem zeroLocus_empty : zeroLocus (∅ : Set R) = Set.univ :=
  (gc_set R).l_bot


@[simp]
theorem vanishingIdeal_empty : vanishingIdeal (∅ : Set (PrimeSpectrum R)) = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (PrimeSpectrum.vanishingIdeal EmptyCollection.emptyCollection) Top.top
  -/
  simpa using (gc R).u_top
  /-
    🎉 no goals
  -/


theorem zeroLocus_empty_of_one_mem {s : Set R} (h : (1 : R) ∈ s) : zeroLocus s = ∅ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    h : Membership.mem s 1
    ⊢ Eq (PrimeSpectrum.zeroLocus s) EmptyCollection.emptyCollection
  -/
  rw [Set.eq_empty_iff_forall_not_mem]
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    h : Membership.mem s 1
    ⊢ ∀ (x : PrimeSpectrum R), Not (Membership.mem (PrimeSpectrum.zeroLocus s) x)
  -/
  intro x hx
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    h : Membership.mem s 1
    x : PrimeSpectrum R
    hx : Membership.mem (PrimeSpectrum.zeroLocus s) x
    ⊢ False
  -/
  rw [mem_zeroLocus] at hx
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    h : Membership.mem s 1
    x : PrimeSpectrum R
    hx : HasSubset.Subset s ↑x.asIdeal
    ⊢ False
  -/
  have x_prime : x.asIdeal.IsPrime := by infer_instance
  have eq_top : x.asIdeal = ⊤ := by
    rw [Ideal.eq_top_iff_one]
    exact hx h
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    h : Membership.mem s 1
    x : PrimeSpectrum R
    hx : HasSubset.Subset s ↑x.asIdeal
    x_prime : x.asIdeal.IsPrime
    eq_top : Eq x.asIdeal Top.top
    ⊢ False
  -/
  apply x_prime.ne_top eq_top
  /-
    🎉 no goals
  -/


@[simp]
theorem zeroLocus_singleton_one : zeroLocus ({1} : Set R) = ∅ :=
  zeroLocus_empty_of_one_mem (Set.mem_singleton (1 : R))


theorem zeroLocus_empty_iff_eq_top {I : Ideal R} : zeroLocus (I : Set R) = ∅ ↔ I = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Iff (Eq (PrimeSpectrum.zeroLocus ↑I) EmptyCollection.emptyCollection) (Eq I  …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      ⊢ Eq (PrimeSpectrum.zeroLocus ↑I) EmptyCollection.emptyCollection → Eq I Top.top
    -/
  · contrapose!
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      ⊢ Ne I Top.top → (PrimeSpectrum.zeroLocus ↑I).Nonempty
    -/
    intro h
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      h : Ne I Top.top
      ⊢ (PrimeSpectrum.zeroLocus ↑I).Nonempty
    -/
    rcases Ideal.exists_le_maximal I h with ⟨M, hM, hIM⟩
    /-
      case mp.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      h : Ne I Top.top
      M : Ideal R
      hM : M.IsMaximal
      hIM : LE.le I M
      ⊢ (PrimeSpectrum.zeroLocus ↑I).Nonempty
    -/
    exact ⟨⟨M, hM.isPrime⟩, hIM⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      ⊢ Eq I Top.top → Eq (PrimeSpectrum.zeroLocus ↑I) EmptyCollection.emptyCollection
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      ⊢ Eq (PrimeSpectrum.zeroLocus ↑Top.top) EmptyCollection.emptyCollection
    -/
    apply zeroLocus_empty_of_one_mem
    /-
      case mpr.h
      R : Type u
      inst✝ : CommSemiring R
      ⊢ Membership.mem (↑Top.top) 1
    -/
    trivial
    /-
      🎉 no goals
    -/


@[simp]
theorem zeroLocus_univ : zeroLocus (Set.univ : Set R) = ∅ :=
  zeroLocus_empty_of_one_mem (Set.mem_univ 1)


theorem vanishingIdeal_eq_top_iff {s : Set (PrimeSpectrum R)} : vanishingIdeal s = ⊤ ↔ s = ∅ := by
  rw [← top_le_iff, ← subset_zeroLocus_iff_le_vanishingIdeal, Submodule.top_coe, zeroLocus_univ,
    Set.subset_empty_iff]


theorem zeroLocus_eq_top_iff (s : Set R) :
    zeroLocus s = ⊤ ↔ s ⊆ nilradical R := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Iff (Eq (PrimeSpectrum.zeroLocus s) Top.top) (HasSubset.Subset s ↑(nilradica …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      s : Set R
      ⊢ Eq (PrimeSpectrum.zeroLocus s) Top.top → HasSubset.Subset s ↑(nilradical R)
    -/
  · intro h x hx
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      s : Set R
      h : Eq (PrimeSpectrum.zeroLocus s) Top.top
      x : R
      hx : Membership.mem s x
      ⊢ Membership.mem (↑(nilradical R)) x
    -/
    refine nilpotent_iff_mem_prime.mpr (fun J hJ ↦ ?_)
    have hJz : ⟨J, hJ⟩ ∈ zeroLocus s := by
      rw [h]
      trivial
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      s : Set R
      h : Eq (PrimeSpectrum.zeroLocus s) Top.top
      x : R
      hx : Membership.mem s x
      J : Ideal R
      hJ : J.IsPrime
      hJz : Membership.mem (PrimeSpectrum.zeroLocus s) { asIdeal := J, isPrime := hJ }
      ⊢ Membership.mem J x
    -/
    exact (mem_zeroLocus _ _).mpr hJz hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      s : Set R
      ⊢ HasSubset.Subset s ↑(nilradical R) → Eq (PrimeSpectrum.zeroLocus s) Top.top
    -/
  · rw [eq_top_iff]
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      s : Set R
      ⊢ HasSubset.Subset s ↑(nilradical R) → LE.le Top.top (PrimeSpectrum.zeroLocus s)
    -/
    intro h p _
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      s : Set R
      h : HasSubset.Subset s ↑(nilradical R)
      p : PrimeSpectrum R
      a✝ : Membership.mem Top.top p
      ⊢ Membership.mem (PrimeSpectrum.zeroLocus s) p
    -/
    apply Set.Subset.trans h (nilradical_le_prime p.asIdeal)
    /-
      🎉 no goals
    -/


theorem zeroLocus_sup (I J : Ideal R) :
    zeroLocus ((I ⊔ J : Ideal R) : Set R) = zeroLocus I ∩ zeroLocus J :=
  (gc R).l_sup


theorem zeroLocus_union (s s' : Set R) : zeroLocus (s ∪ s') = zeroLocus s ∩ zeroLocus s' :=
  (gc_set R).l_sup


theorem vanishingIdeal_union (t t' : Set (PrimeSpectrum R)) :
    vanishingIdeal (t ∪ t') = vanishingIdeal t ⊓ vanishingIdeal t' :=
  (gc R).u_inf


theorem zeroLocus_iSup {ι : Sort*} (I : ι → Ideal R) :
    zeroLocus ((⨆ i, I i : Ideal R) : Set R) = ⋂ i, zeroLocus (I i) :=
  (gc R).l_iSup


theorem zeroLocus_iUnion {ι : Sort*} (s : ι → Set R) :
    zeroLocus (⋃ i, s i) = ⋂ i, zeroLocus (s i) :=
  (gc_set R).l_iSup


theorem zeroLocus_iUnion₂ {ι : Sort*} {κ : (i : ι) → Sort*} (s : ∀ i, κ i → Set R) :
    zeroLocus (⋃ (i) (j), s i j) = ⋂ (i) (j), zeroLocus (s i j) :=
  (gc_set R).l_iSup₂


theorem zeroLocus_bUnion (s : Set (Set R)) :
                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : CommSemiring R
                                                                      s : Set (Set R)
                                                                      ⊢ Eq (PrimeSpectrum.zeroLocus (Set.iUnion fun s' => Set.iUnion fun h => s')) ( …
                                                                    -/
    zeroLocus (⋃ s' ∈ s, s' : Set R) = ⋂ s' ∈ s, zeroLocus s' := by simp only [zeroLocus_iUnion]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem vanishingIdeal_iUnion {ι : Sort*} (t : ι → Set (PrimeSpectrum R)) :
    vanishingIdeal (⋃ i, t i) = ⨅ i, vanishingIdeal (t i) :=
  (gc R).u_iInf


theorem zeroLocus_inf (I J : Ideal R) :
    zeroLocus ((I ⊓ J : Ideal R) : Set R) = zeroLocus I ∪ zeroLocus J :=
  Set.ext fun x => x.2.inf_le


theorem union_zeroLocus (s s' : Set R) :
    zeroLocus s ∪ zeroLocus s' = zeroLocus (Ideal.span s ⊓ Ideal.span s' : Ideal R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s s' : Set R
    ⊢ Eq (Union.union (PrimeSpectrum.zeroLocus s) (PrimeSpectrum.zeroLocus s')) (P …
  -/
  rw [zeroLocus_inf]
  /-
    R : Type u
    inst✝ : CommSemiring R
    s s' : Set R
    ⊢ Eq (Union.union (PrimeSpectrum.zeroLocus s) (PrimeSpectrum.zeroLocus s')) (U …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem zeroLocus_mul (I J : Ideal R) :
    zeroLocus ((I * J : Ideal R) : Set R) = zeroLocus I ∪ zeroLocus J :=
  Set.ext fun x => x.2.mul_le


theorem zeroLocus_singleton_mul (f g : R) :
    zeroLocus ({f * g} : Set R) = zeroLocus {f} ∪ zeroLocus {g} :=
                      /-
                        R : Type u
                        inst✝ : CommSemiring R
                        f g : R
                        x : PrimeSpectrum R
                        ⊢ Iff (Membership.mem (PrimeSpectrum.zeroLocus (Singleton.singleton (HMul.hMul …
                      -/
  Set.ext fun x => by simpa using x.2.mul_mem_iff_mem_or_mem
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem zeroLocus_pow (I : Ideal R) {n : ℕ} (hn : n ≠ 0) :
    zeroLocus ((I ^ n : Ideal R) : Set R) = zeroLocus I :=
  zeroLocus_radical (I ^ n) ▸ (I.radical_pow hn).symm ▸ zeroLocus_radical I


@[simp]
theorem zeroLocus_singleton_pow (f : R) (n : ℕ) (hn : 0 < n) :
    zeroLocus ({f ^ n} : Set R) = zeroLocus {f} :=
                      /-
                        R : Type u
                        inst✝ : CommSemiring R
                        f : R
                        n : Nat
                        hn : LT.lt 0 n
                        x : PrimeSpectrum R
                        ⊢ Iff (Membership.mem (PrimeSpectrum.zeroLocus (Singleton.singleton (HPow.hPow …
                      -/
  Set.ext fun x => by simpa using x.2.pow_mem_iff_mem n hn
                      /-
                        🎉 no goals
                      -/


theorem sup_vanishingIdeal_le (t t' : Set (PrimeSpectrum R)) :
    vanishingIdeal t ⊔ vanishingIdeal t' ≤ vanishingIdeal (t ∩ t') := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t t' : Set (PrimeSpectrum R)
    ⊢ LE.le (Max.max (PrimeSpectrum.vanishingIdeal t) (PrimeSpectrum.vanishingIdea …
  -/
  intro r
  /-
    R : Type u
    inst✝ : CommSemiring R
    t t' : Set (PrimeSpectrum R)
    r : R
    ⊢ Membership.mem (Max.max (PrimeSpectrum.vanishingIdeal t) (PrimeSpectrum.vani …
  -/
  rw [Submodule.mem_sup, mem_vanishingIdeal]
  /-
    R : Type u
    inst✝ : CommSemiring R
    t t' : Set (PrimeSpectrum R)
    r : R
    ⊢ (Exists fun y => And (Membership.mem (PrimeSpectrum.vanishingIdeal t) y) (Ex …
  -/
  rintro ⟨f, hf, g, hg, rfl⟩ x ⟨hxt, hxt'⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    t t' : Set (PrimeSpectrum R)
    f : R
    hf : Membership.mem (PrimeSpectrum.vanishingIdeal t) f
    g : R
    hg : Membership.mem (PrimeSpectrum.vanishingIdeal t') g
    x : PrimeSpectrum R
    hxt : Membership.mem t x
    hxt' : Membership.mem t' x
    ⊢ Membership.mem x.asIdeal (HAdd.hAdd f g)
  -/
  rw [mem_vanishingIdeal] at hf hg
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    t t' : Set (PrimeSpectrum R)
    f : R
    hf : ∀ (x : PrimeSpectrum R), Membership.mem t x → Membership.mem x.asIdeal f
    g : R
    hg : ∀ (x : PrimeSpectrum R), Membership.mem t' x → Membership.mem x.asIdeal g
    x : PrimeSpectrum R
    hxt : Membership.mem t x
    hxt' : Membership.mem t' x
    ⊢ Membership.mem x.asIdeal (HAdd.hAdd f g)
  -/
                              /-
                                🎉 no goals
                              -/
  apply Submodule.add_mem <;> solve_by_elim
                              /-
                                🎉 no goals
                              -/


theorem mem_compl_zeroLocus_iff_not_mem {f : R} {I : PrimeSpectrum R} :
    I ∈ (zeroLocus {f} : Set (PrimeSpectrum R))ᶜ ↔ f ∉ I.asIdeal := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f : R
    I : PrimeSpectrum R
    ⊢ Iff (Membership.mem (HasCompl.compl (PrimeSpectrum.zeroLocus (Singleton.sing …
  -/
  rw [Set.mem_compl_iff, mem_zeroLocus, Set.singleton_subset_iff]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
lemma zeroLocus_insert_zero (s : Set R) : zeroLocus (insert 0 s) = zeroLocus s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Eq (PrimeSpectrum.zeroLocus (Insert.insert 0 s)) (PrimeSpectrum.zeroLocus s)
  -/
  rw [← Set.union_singleton, zeroLocus_union, zeroLocus_singleton_zero, Set.inter_univ]
  /-
    🎉 no goals
  -/


@[simp]
lemma zeroLocus_diff_singleton_zero (s : Set R) : zeroLocus (s \ {0}) = zeroLocus s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    s : Set R
    ⊢ Eq (PrimeSpectrum.zeroLocus (SDiff.sdiff s (Singleton.singleton 0))) (PrimeS …
  -/
  rw [← zeroLocus_insert_zero, ← zeroLocus_insert_zero (s := s)]; simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma zeroLocus_smul_of_isUnit {r : R} (hr : IsUnit r) (s : Set R) :
    zeroLocus (r • s) = zeroLocus s := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    r : R
    hr : IsUnit r
    s : Set R
    ⊢ Eq (PrimeSpectrum.zeroLocus (HSMul.hSMul r s)) (PrimeSpectrum.zeroLocus s)
  -/
  ext; simp [Set.subset_def, ← Set.image_smul, Ideal.unit_mul_mem_iff_mem _ hr]
       /-
         🎉 no goals
       -/


instance : PartialOrder (PrimeSpectrum R) :=
  PartialOrder.lift asIdeal (@PrimeSpectrum.ext _ _)


@[simp]
theorem asIdeal_le_asIdeal (x y : PrimeSpectrum R) : x.asIdeal ≤ y.asIdeal ↔ x ≤ y :=
  Iff.rfl


@[simp]
theorem asIdeal_lt_asIdeal (x y : PrimeSpectrum R) : x.asIdeal < y.asIdeal ↔ x < y :=
  Iff.rfl


instance [IsDomain R] : OrderBot (PrimeSpectrum R) where
  bot := ⟨⊥, Ideal.bot_prime⟩
  bot_le I := @bot_le _ _ _ I.asIdeal


instance {R : Type*} [Field R] : Unique (PrimeSpectrum R) where
  default := ⊥
  uniq x := PrimeSpectrum.ext ((IsSimpleOrder.eq_bot_or_eq_top _).resolve_right x.2.ne_top)


/-- In a noetherian ring, every ideal contains a product of prime ideals
([samuel, § 3.3, Lemma 3])-/
theorem exists_primeSpectrum_prod_le (I : Ideal R) :
    ∃ Z : Multiset (PrimeSpectrum R), Multiset.prod (Z.map asIdeal) ≤ I := by
  -- Porting note: Need to specify `P` explicitly
  refine IsNoetherian.induction
    (P := fun I => ∃ Z : Multiset (PrimeSpectrum R), Multiset.prod (Z.map asIdeal) ≤ I)
    (fun (M : Ideal R) hgt => ?_) I
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
  -/
  by_cases h_prM : M.IsPrime
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I M : Ideal R
      hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
      h_prM : M.IsPrime
      ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
    -/
  · use {⟨M, h_prM⟩}
    /-
      case h
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I M : Ideal R
      hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
      h_prM : M.IsPrime
      ⊢ LE.le (Multiset.map PrimeSpectrum.asIdeal (Singleton.singleton { asIdeal :=  …
    -/
    rw [Multiset.map_singleton, Multiset.prod_singleton]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
  -/
  by_cases htop : M = ⊤
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I M : Ideal R
      hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
      h_prM : Not M.IsPrime
      htop : Eq M Top.top
      ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
    -/
  · rw [htop]
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I M : Ideal R
      hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
      h_prM : Not M.IsPrime
      htop : Eq M Top.top
      ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
    -/
    exact ⟨0, le_top⟩
    /-
      🎉 no goals
    -/
  have lt_add : ∀ z ∉ M, M < M + span R {z} := by
    intro z hz
    refine lt_of_le_of_ne le_sup_left fun m_eq => hz ?_
    rw [m_eq]
    exact Ideal.mem_sup_right (mem_span_singleton_self z)
  /-
    case neg
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
  -/
  obtain ⟨x, hx, y, hy, hxy⟩ := (Ideal.not_isPrime_iff.mp h_prM).resolve_left htop
  /-
    case neg.intro.intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
  -/
  obtain ⟨Wx, h_Wx⟩ := hgt (M + span R {x}) (lt_add _ hx)
  /-
    case neg.intro.intro.intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
  -/
  obtain ⟨Wy, h_Wy⟩ := hgt (M + span R {y}) (lt_add _ hy)
  /-
    case neg.intro.intro.intro.intro.intro.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ (fun I => Exists fun Z => LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod  …
  -/
  use Wx + Wy
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (Multiset.map PrimeSpectrum.asIdeal (HAdd.hAdd Wx Wy)).prod M
  -/
  rw [Multiset.map_add, Multiset.prod_add]
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (HMul.hMul (Multiset.map PrimeSpectrum.asIdeal Wx).prod (Multiset.map  …
  -/
  apply le_trans (Submodule.mul_le_mul h_Wx h_Wy)
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (HMul.hMul (HAdd.hAdd M (Submodule.span R (Singleton.singleton x))) (H …
  -/
  rw [add_mul]
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul M (HAdd.hAdd M (Submodule.span R (Singleton.sing …
  -/
  apply sup_le (show M * (M + span R {y}) ≤ M from Ideal.mul_le_right)
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (HMul.hMul (Submodule.span R (Singleton.singleton x)) (HAdd.hAdd M (Su …
  -/
  rw [mul_add]
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (Submodule.span R (Singleton.singleton x)) M) (H …
  -/
  apply sup_le (show span R {x} * M ≤ M from Ideal.mul_le_left)
  /-
    case h
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I M : Ideal R
    hgt : ∀ (J : Submodule R R), GT.gt J M → (fun I => Exists fun Z => LE.le (Mult …
    h_prM : Not M.IsPrime
    htop : Not (Eq M Top.top)
    lt_add : ∀ (z : R), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    x : R
    hx : Not (Membership.mem M x)
    y : R
    hy : Not (Membership.mem M y)
    hxy : Membership.mem M (HMul.hMul x y)
    Wx : Multiset (PrimeSpectrum R)
    h_Wx : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Submod …
    Wy : Multiset (PrimeSpectrum R)
    h_Wy : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Submod …
    ⊢ LE.le (HMul.hMul (Submodule.span R (Singleton.singleton x)) (Submodule.span  …
  -/
  rwa [span_mul_span, Set.singleton_mul_singleton, span_singleton_le_iff_mem]
  /-
    🎉 no goals
  -/


/-- In a noetherian integral domain which is not a field, every non-zero ideal contains a non-zero
  product of prime ideals; in a field, the whole ring is a non-zero ideal containing only 0 as
  product or prime ideals ([samuel, § 3.3, Lemma 3]) -/
theorem exists_primeSpectrum_prod_le_and_ne_bot_of_domain (h_fA : ¬IsField A) {I : Ideal A}
    (h_nzI : I ≠ ⊥) :
    ∃ Z : Multiset (PrimeSpectrum A),
      Multiset.prod (Z.map asIdeal) ≤ I ∧ Multiset.prod (Z.map asIdeal) ≠ ⊥ := by
  /-
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I : Ideal A
    h_nzI : Ne I Bot.bot
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod I) (N …
  -/
  revert h_nzI
  -- Porting note: Need to specify `P` explicitly
  refine IsNoetherian.induction (P := fun I => I ≠ ⊥ → ∃ Z : Multiset (PrimeSpectrum A),
      Multiset.prod (Z.map asIdeal) ≤ I ∧ Multiset.prod (Z.map asIdeal) ≠ ⊥)
    (fun (M : Ideal A) hgt => ?_) I
  /-
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    ⊢ (fun I => Ne I Bot.bot → Exists fun Z => And (LE.le (Multiset.map PrimeSpect …
  -/
  intro h_nzM
  /-
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  have hA_nont : Nontrivial A := IsDomain.toNontrivial
  /-
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  by_cases h_topM : M = ⊤
    /-
      case pos
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Eq M Top.top
      ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
    -/
  · rcases h_topM with rfl
    obtain ⟨p_id, h_nzp, h_pp⟩ : ∃ p : Ideal A, p ≠ ⊥ ∧ p.IsPrime := by
      apply Ring.not_isField_iff_exists_prime.mp h_fA
    /-
      case pos.intro.intro
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I : Ideal A
      hA_nont : Nontrivial A
      hgt : ∀ (J : Submodule A A), GT.gt J Top.top → (fun I => Ne I Bot.bot → Exists …
      h_nzM : Ne Top.top Bot.bot
      p_id : Ideal A
      h_nzp : Ne p_id Bot.bot
      h_pp : p_id.IsPrime
      ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod Top.t …
    -/
    use ({⟨p_id, h_pp⟩} : Multiset (PrimeSpectrum A)), le_top
    /-
      case right
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I : Ideal A
      hA_nont : Nontrivial A
      hgt : ∀ (J : Submodule A A), GT.gt J Top.top → (fun I => Ne I Bot.bot → Exists …
      h_nzM : Ne Top.top Bot.bot
      p_id : Ideal A
      h_nzp : Ne p_id Bot.bot
      h_pp : p_id.IsPrime
      ⊢ Ne (Multiset.map PrimeSpectrum.asIdeal (Singleton.singleton { asIdeal := p_i …
    -/
    rwa [Multiset.map_singleton, Multiset.prod_singleton]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  by_cases h_prM : M.IsPrime
    /-
      case pos
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : M.IsPrime
      ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
    -/
  · use ({⟨M, h_prM⟩} : Multiset (PrimeSpectrum A))
    /-
      case h
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : M.IsPrime
      ⊢ And (LE.le (Multiset.map PrimeSpectrum.asIdeal (Singleton.singleton { asIdea …
    -/
    rw [Multiset.map_singleton, Multiset.prod_singleton]
    /-
      case h
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : M.IsPrime
      ⊢ And (LE.le { asIdeal := M, isPrime := h_prM }.asIdeal M) (Ne { asIdeal := M, …
    -/
    exact ⟨le_rfl, h_nzM⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    h_prM : Not M.IsPrime
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  obtain ⟨x, hx, y, hy, h_xy⟩ := (Ideal.not_isPrime_iff.mp h_prM).resolve_left h_topM
  have lt_add : ∀ z ∉ M, M < M + span A {z} := by
    intro z hz
    refine lt_of_le_of_ne le_sup_left fun m_eq => hz ?_
    rw [m_eq]
    exact mem_sup_right (mem_span_singleton_self z)
  /-
    case neg.intro.intro.intro.intro
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    h_prM : Not M.IsPrime
    x : A
    hx : Not (Membership.mem M x)
    y : A
    hy : Not (Membership.mem M y)
    h_xy : Membership.mem M (HMul.hMul x y)
    lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  obtain ⟨Wx, h_Wx_le, h_Wx_ne⟩ := hgt (M + span A {x}) (lt_add _ hx) (ne_bot_of_gt (lt_add _ hx))
  /-
    case neg.intro.intro.intro.intro.intro.intro
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    h_prM : Not M.IsPrime
    x : A
    hx : Not (Membership.mem M x)
    y : A
    hy : Not (Membership.mem M y)
    h_xy : Membership.mem M (HMul.hMul x y)
    lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    Wx : Multiset (PrimeSpectrum A)
    h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
    h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  obtain ⟨Wy, h_Wy_le, h_Wx_ne⟩ := hgt (M + span A {y}) (lt_add _ hy) (ne_bot_of_gt (lt_add _ hy))
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    h_prM : Not M.IsPrime
    x : A
    hx : Not (Membership.mem M x)
    y : A
    hy : Not (Membership.mem M y)
    h_xy : Membership.mem M (HMul.hMul x y)
    lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    Wx : Multiset (PrimeSpectrum A)
    h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
    h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
    Wy : Multiset (PrimeSpectrum A)
    h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
    h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
    ⊢ Exists fun Z => And (LE.le (Multiset.map PrimeSpectrum.asIdeal Z).prod M) (N …
  -/
  use Wx + Wy
  /-
    case h
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    h_prM : Not M.IsPrime
    x : A
    hx : Not (Membership.mem M x)
    y : A
    hy : Not (Membership.mem M y)
    h_xy : Membership.mem M (HMul.hMul x y)
    lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    Wx : Multiset (PrimeSpectrum A)
    h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
    h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
    Wy : Multiset (PrimeSpectrum A)
    h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
    h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
    ⊢ And (LE.le (Multiset.map PrimeSpectrum.asIdeal (HAdd.hAdd Wx Wy)).prod M) (N …
  -/
  rw [Multiset.map_add, Multiset.prod_add]
  /-
    case h
    A : Type u
    inst✝² : CommRing A
    inst✝¹ : IsDomain A
    inst✝ : IsNoetherianRing A
    h_fA : Not (IsField A)
    I M : Ideal A
    hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
    h_nzM : Ne M Bot.bot
    hA_nont : Nontrivial A
    h_topM : Not (Eq M Top.top)
    h_prM : Not M.IsPrime
    x : A
    hx : Not (Membership.mem M x)
    y : A
    hy : Not (Membership.mem M y)
    h_xy : Membership.mem M (HMul.hMul x y)
    lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
    Wx : Multiset (PrimeSpectrum A)
    h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
    h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
    Wy : Multiset (PrimeSpectrum A)
    h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
    h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
    ⊢ And (LE.le (HMul.hMul (Multiset.map PrimeSpectrum.asIdeal Wx).prod (Multiset …
  -/
  refine ⟨le_trans (Submodule.mul_le_mul h_Wx_le h_Wy_le) ?_, mt Ideal.mul_eq_bot.mp ?_⟩
    /-
      case h.refine_1
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : Not M.IsPrime
      x : A
      hx : Not (Membership.mem M x)
      y : A
      hy : Not (Membership.mem M y)
      h_xy : Membership.mem M (HMul.hMul x y)
      lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
      Wx : Multiset (PrimeSpectrum A)
      h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
      h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
      Wy : Multiset (PrimeSpectrum A)
      h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
      h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
      ⊢ LE.le (HMul.hMul (HAdd.hAdd M (Submodule.span A (Singleton.singleton x))) (H …
    -/
  · rw [add_mul]
    /-
      case h.refine_1
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : Not M.IsPrime
      x : A
      hx : Not (Membership.mem M x)
      y : A
      hy : Not (Membership.mem M y)
      h_xy : Membership.mem M (HMul.hMul x y)
      lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
      Wx : Multiset (PrimeSpectrum A)
      h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
      h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
      Wy : Multiset (PrimeSpectrum A)
      h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
      h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
      ⊢ LE.le (HAdd.hAdd (HMul.hMul M (HAdd.hAdd M (Submodule.span A (Singleton.sing …
    -/
    apply sup_le (show M * (M + span A {y}) ≤ M from Ideal.mul_le_right)
    /-
      case h.refine_1
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : Not M.IsPrime
      x : A
      hx : Not (Membership.mem M x)
      y : A
      hy : Not (Membership.mem M y)
      h_xy : Membership.mem M (HMul.hMul x y)
      lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
      Wx : Multiset (PrimeSpectrum A)
      h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
      h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
      Wy : Multiset (PrimeSpectrum A)
      h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
      h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
      ⊢ LE.le (HMul.hMul (Submodule.span A (Singleton.singleton x)) (HAdd.hAdd M (Su …
    -/
    rw [mul_add]
    /-
      case h.refine_1
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : Not M.IsPrime
      x : A
      hx : Not (Membership.mem M x)
      y : A
      hy : Not (Membership.mem M y)
      h_xy : Membership.mem M (HMul.hMul x y)
      lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
      Wx : Multiset (PrimeSpectrum A)
      h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
      h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
      Wy : Multiset (PrimeSpectrum A)
      h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
      h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (Submodule.span A (Singleton.singleton x)) M) (H …
    -/
    apply sup_le (show span A {x} * M ≤ M from Ideal.mul_le_left)
    /-
      case h.refine_1
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : Not M.IsPrime
      x : A
      hx : Not (Membership.mem M x)
      y : A
      hy : Not (Membership.mem M y)
      h_xy : Membership.mem M (HMul.hMul x y)
      lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
      Wx : Multiset (PrimeSpectrum A)
      h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
      h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
      Wy : Multiset (PrimeSpectrum A)
      h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
      h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
      ⊢ LE.le (HMul.hMul (Submodule.span A (Singleton.singleton x)) (Submodule.span  …
    -/
    rwa [span_mul_span, Set.singleton_mul_singleton, span_singleton_le_iff_mem]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      A : Type u
      inst✝² : CommRing A
      inst✝¹ : IsDomain A
      inst✝ : IsNoetherianRing A
      h_fA : Not (IsField A)
      I M : Ideal A
      hgt : ∀ (J : Submodule A A), GT.gt J M → (fun I => Ne I Bot.bot → Exists fun Z …
      h_nzM : Ne M Bot.bot
      hA_nont : Nontrivial A
      h_topM : Not (Eq M Top.top)
      h_prM : Not M.IsPrime
      x : A
      hx : Not (Membership.mem M x)
      y : A
      hy : Not (Membership.mem M y)
      h_xy : Membership.mem M (HMul.hMul x y)
      lt_add : ∀ (z : A), Not (Membership.mem M z) → LT.lt M (HAdd.hAdd M (Submodule …
      Wx : Multiset (PrimeSpectrum A)
      h_Wx_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wx).prod (HAdd.hAdd M (Sub …
      h_Wx_ne✝ : Ne (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot
      Wy : Multiset (PrimeSpectrum A)
      h_Wy_le : LE.le (Multiset.map PrimeSpectrum.asIdeal Wy).prod (HAdd.hAdd M (Sub …
      h_Wx_ne : Ne (Multiset.map PrimeSpectrum.asIdeal Wy).prod Bot.bot
      ⊢ Not (Or (Eq (Multiset.map PrimeSpectrum.asIdeal Wx).prod Bot.bot) (Eq (Multi …
    -/
                         /-
                           🎉 no goals
                         -/
  · rintro (hx | hy) <;> contradiction
                         /-
                           🎉 no goals
                         -/


/-- The pullback of an element of `PrimeSpectrum S` along a ring homomorphism `f : R →+* S`.
The bundled continuous version is `PrimeSpectrum.comap`. -/
abbrev RingHom.specComap {R S : Type*} [CommSemiring R] [CommSemiring S] (f : R →+* S) :
    PrimeSpectrum S → PrimeSpectrum R :=
  fun y => ⟨Ideal.comap f y.asIdeal, inferInstance⟩


theorem preimage_specComap_zeroLocus_aux (f : R →+* S) (s : Set R) :
    f.specComap ⁻¹' zeroLocus s = zeroLocus (f '' s) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    s : Set R
    ⊢ Eq (Set.preimage f.specComap (PrimeSpectrum.zeroLocus s)) (PrimeSpectrum.zer …
  -/
  ext x
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    s : Set R
    x : PrimeSpectrum S
    ⊢ Iff (Membership.mem (Set.preimage f.specComap (PrimeSpectrum.zeroLocus s)) x …
  -/
  simp only [mem_zeroLocus, Set.image_subset_iff, Set.mem_preimage, mem_zeroLocus, Ideal.coe_comap]
  /-
    🎉 no goals
  -/


@[simp]
theorem specComap_asIdeal (y : PrimeSpectrum S) :
    (f.specComap y).asIdeal = Ideal.comap f y.asIdeal :=
  rfl


@[simp]
theorem specComap_id : (RingHom.id R).specComap = fun x => x :=
  rfl


@[simp]
theorem specComap_comp (f : R →+* S) (g : S →+* S') :
    (g.comp f).specComap = f.specComap.comp g.specComap :=
  rfl


theorem specComap_comp_apply (f : R →+* S) (g : S →+* S') (x : PrimeSpectrum S') :
    (g.comp f).specComap x = f.specComap (g.specComap x) :=
  rfl


@[simp]
theorem preimage_specComap_zeroLocus (s : Set R) :
    f.specComap ⁻¹' zeroLocus s = zeroLocus (f '' s) :=
  preimage_specComap_zeroLocus_aux f s


theorem specComap_injective_of_surjective (f : R →+* S) (hf : Function.Surjective f) :
    Function.Injective f.specComap := fun x y h =>
  PrimeSpectrum.ext
    (Ideal.comap_injective_of_surjective f hf
      (congr_arg PrimeSpectrum.asIdeal h : (f.specComap x).asIdeal = (f.specComap y).asIdeal))


/-- `RingHom.specComap` of an isomorphism of rings as an equivalence of their prime spectra. -/
@[simps apply symm_apply]
def comapEquiv (e : R ≃+* S) : PrimeSpectrum R ≃ PrimeSpectrum S where
  toFun := e.symm.toRingHom.specComap
  invFun := e.toRingHom.specComap
  left_inv x := by
    rw [← specComap_comp_apply, RingEquiv.toRingHom_eq_coe,
      RingEquiv.toRingHom_eq_coe, RingEquiv.symm_comp]
    /-
      R : Type u
      S : Type v
      S' : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : CommSemiring S'
      f : RingHom R S
      e : RingEquiv R S
      x : PrimeSpectrum R
      ⊢ Eq ((RingHom.id R).specComap x) x
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv x := by
    rw [← specComap_comp_apply, RingEquiv.toRingHom_eq_coe,
      RingEquiv.toRingHom_eq_coe, RingEquiv.comp_symm]
    /-
      R : Type u
      S : Type v
      S' : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : CommSemiring S'
      f : RingHom R S
      e : RingEquiv R S
      x : PrimeSpectrum S
      ⊢ Eq ((RingHom.id S).specComap x) x
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem localization_specComap_injective [Algebra R S] (M : Submonoid R) [IsLocalization M S] :
    Function.Injective (algebraMap R S).specComap := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    ⊢ Function.Injective (algebraMap R S).specComap
  -/
  intro p q h
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    p q : PrimeSpectrum S
    h : Eq ((algebraMap R S).specComap p) ((algebraMap R S).specComap q)
    ⊢ Eq p q
  -/
  replace h := _root_.congr_arg (fun x : PrimeSpectrum R => Ideal.map (algebraMap R S) x.asIdeal) h
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    p q : PrimeSpectrum S
    h : Eq ((fun x => Ideal.map (algebraMap R S) x.asIdeal) ((algebraMap R S).spec …
    ⊢ Eq p q
  -/
  dsimp only [specComap] at h
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    p q : PrimeSpectrum S
    h : Eq (Ideal.map (algebraMap R S) (Ideal.comap (algebraMap R S) p.asIdeal)) ( …
    ⊢ Eq p q
  -/
  rw [IsLocalization.map_comap M S, IsLocalization.map_comap M S] at h
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    p q : PrimeSpectrum S
    h : Eq p.asIdeal q.asIdeal
    ⊢ Eq p q
  -/
  ext1
  /-
    case asIdeal
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    p q : PrimeSpectrum S
    h : Eq p.asIdeal q.asIdeal
    ⊢ Eq p.asIdeal q.asIdeal
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem localization_specComap_range [Algebra R S] (M : Submonoid R) [IsLocalization M S] :
    Set.range (algebraMap R S).specComap = { p | Disjoint (M : Set R) p.asIdeal } := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    ⊢ Eq (Set.range (algebraMap R S).specComap) (setOf fun p => Disjoint ↑M ↑p.asI …
  -/
  ext x
  /-
    case h
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    x : PrimeSpectrum R
    ⊢ Iff (Membership.mem (Set.range (algebraMap R S).specComap) x) (Membership.me …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : PrimeSpectrum R
      ⊢ Membership.mem (Set.range (algebraMap R S).specComap) x → Membership.mem (se …
    -/
  · simp_rw [disjoint_iff_inf_le]
    /-
      case h.mp
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : PrimeSpectrum R
      ⊢ Membership.mem (Set.range (algebraMap R S).specComap) x → Membership.mem (se …
    -/
    rintro ⟨p, rfl⟩ x ⟨hx₁, hx₂⟩
    /-
      case h.mp.intro.intro
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      p : PrimeSpectrum S
      x : R
      hx₁ : Membership.mem (↑M) x
      hx₂ : Membership.mem (↑((algebraMap R S).specComap p).asIdeal) x
      ⊢ Membership.mem Bot.bot x
    -/
    exact (p.2.1 : ¬_) (p.asIdeal.eq_top_of_isUnit_mem hx₂ (IsLocalization.map_units S ⟨x, hx₁⟩))
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : PrimeSpectrum R
      ⊢ Membership.mem (setOf fun p => Disjoint ↑M ↑p.asIdeal) x → Membership.mem (S …
    -/
  · intro h
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : PrimeSpectrum R
      h : Membership.mem (setOf fun p => Disjoint ↑M ↑p.asIdeal) x
      ⊢ Membership.mem (Set.range (algebraMap R S).specComap) x
    -/
    use ⟨x.asIdeal.map (algebraMap R S), IsLocalization.isPrime_of_isPrime_disjoint M S _ x.2 h⟩
    /-
      case h
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : PrimeSpectrum R
      h : Membership.mem (setOf fun p => Disjoint ↑M ↑p.asIdeal) x
      ⊢ Eq ((algebraMap R S).specComap { asIdeal := Ideal.map (algebraMap R S) x.asI …
    -/
    ext1
    /-
      case h.asIdeal
      R : Type u
      S : Type v
      inst✝³ : CommSemiring R
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      M : Submonoid R
      inst✝ : IsLocalization M S
      x : PrimeSpectrum R
      h : Membership.mem (setOf fun p => Disjoint ↑M ↑p.asIdeal) x
      ⊢ Eq ((algebraMap R S).specComap { asIdeal := Ideal.map (algebraMap R S) x.asI …
    -/
    exact IsLocalization.comap_map_of_isPrime_disjoint M S _ x.2 h
    /-
      🎉 no goals
    -/


/-- The canonical map from a disjoint union of prime spectra of commutative semirings to
the prime spectrum of the product semiring. -/
/- TODO: show this is always a topological embedding (even when ι is infinite)
and is a homeomorphism when ι is finite. -/
@[simps] def sigmaToPi : (Σ i, PrimeSpectrum (R i)) → PrimeSpectrum (Π i, R i)
  | ⟨i, p⟩ => (Pi.evalRingHom R i).specComap p


theorem sigmaToPi_injective : (sigmaToPi R).Injective := fun ⟨i, p⟩ ⟨j, q⟩ eq ↦ by
  /-
    ι : Type u_3
    R : ι → Type u_2
    inst✝ : (i : ι) → CommSemiring (R i)
    x✝¹ x✝ : Sigma fun i => PrimeSpectrum (R i)
    i : ι
    p : PrimeSpectrum (R i)
    j : ι
    q : PrimeSpectrum (R j)
    eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) (PrimeSpectrum.sigmaToPi R ⟨j, q⟩)
    ⊢ Eq ⟨i, p⟩ ⟨j, q⟩
  -/
  obtain rfl | ne := eq_or_ne i j
    /-
      case inl
      ι : Type u_3
      R : ι → Type u_2
      inst✝ : (i : ι) → CommSemiring (R i)
      x✝¹ x✝ : Sigma fun i => PrimeSpectrum (R i)
      i : ι
      p q : PrimeSpectrum (R i)
      eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) (PrimeSpectrum.sigmaToPi R ⟨i, q⟩)
      ⊢ Eq ⟨i, p⟩ ⟨i, q⟩
    -/
  · congr; ext x
    /-
      case inl.e_snd.asIdeal.h
      ι : Type u_3
      R : ι → Type u_2
      inst✝ : (i : ι) → CommSemiring (R i)
      x✝¹ x✝ : Sigma fun i => PrimeSpectrum (R i)
      i : ι
      p q : PrimeSpectrum (R i)
      eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) (PrimeSpectrum.sigmaToPi R ⟨i, q⟩)
      x : R i
      ⊢ Iff (Membership.mem p.asIdeal x) (Membership.mem q.asIdeal x)
    -/
    simpa using congr_arg (Function.update (0 : ∀ i, R i) i x ∈ ·.asIdeal) eq
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_3
      R : ι → Type u_2
      inst✝ : (i : ι) → CommSemiring (R i)
      x✝¹ x✝ : Sigma fun i => PrimeSpectrum (R i)
      i : ι
      p : PrimeSpectrum (R i)
      j : ι
      q : PrimeSpectrum (R j)
      eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) (PrimeSpectrum.sigmaToPi R ⟨j, q⟩)
      ne : Ne i j
      ⊢ Eq ⟨i, p⟩ ⟨j, q⟩
    -/
  · refine (p.1.ne_top_iff_one.mp p.2.ne_top ?_).elim
    /-
      case inr
      ι : Type u_3
      R : ι → Type u_2
      inst✝ : (i : ι) → CommSemiring (R i)
      x✝¹ x✝ : Sigma fun i => PrimeSpectrum (R i)
      i : ι
      p : PrimeSpectrum (R i)
      j : ι
      q : PrimeSpectrum (R j)
      eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) (PrimeSpectrum.sigmaToPi R ⟨j, q⟩)
      ne : Ne i j
      ⊢ Membership.mem p.asIdeal 1
    -/
    have : Function.update (1 : ∀ i, R i) j 0 ∈ (sigmaToPi R ⟨j, q⟩).asIdeal := by simp
    /-
      case inr
      ι : Type u_3
      R : ι → Type u_2
      inst✝ : (i : ι) → CommSemiring (R i)
      x✝¹ x✝ : Sigma fun i => PrimeSpectrum (R i)
      i : ι
      p : PrimeSpectrum (R i)
      j : ι
      q : PrimeSpectrum (R j)
      eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) (PrimeSpectrum.sigmaToPi R ⟨j, q⟩)
      ne : Ne i j
      this : Membership.mem (PrimeSpectrum.sigmaToPi R ⟨j, q⟩).asIdeal (Function.upd …
      ⊢ Membership.mem p.asIdeal 1
    -/
    simpa [← eq, Function.update_of_ne ne]
    /-
      🎉 no goals
    -/


/-- An infinite product of nontrivial commutative semirings has a maximal ideal outside of the
range of `sigmaToPi`, i.e. is not of the form `πᵢ⁻¹(𝔭)` for some prime `𝔭 ⊂ R i`, where
`πᵢ : (Π i, R i) →+* R i` is the projection. For a complete description of all prime ideals,
see https://math.stackexchange.com/a/1563190. -/
theorem exists_maximal_nmem_range_sigmaToPi_of_infinite :
    ∃ (I : Ideal (Π i, R i)) (_ : I.IsMaximal), ⟨I, inferInstance⟩ ∉ Set.range (sigmaToPi R) := by
  let J : Ideal (Π i, R i) := -- `J := Π₀ i, R i` is an ideal in `Π i, R i`
  { __ := AddMonoidHom.mrange DFinsupp.coeFnAddMonoidHom
    smul_mem' := by
      rintro r _ ⟨x, rfl⟩
      refine ⟨.mk x.support fun i ↦ r i * x i, funext fun i ↦ show dite _ _ _ = _ from ?_⟩
      simp_rw [DFinsupp.coeFnAddMonoidHom]
      refine dite_eq_left_iff.mpr fun h ↦ ?_
      rw [DFinsupp.not_mem_support_iff.mp h, mul_zero] }
  have ⟨I, max, le⟩ := J.exists_le_maximal <| (Ideal.ne_top_iff_one _).mpr <| by
    -- take a maximal ideal I containing J
    rintro ⟨x, hx⟩
    have ⟨i, hi⟩ := x.support.exists_not_mem
    simpa [DFinsupp.coeFnAddMonoidHom, DFinsupp.not_mem_support_iff.mp hi] using congr_fun hx i
  /-
    ι : Type u_3
    R : ι → Type u_2
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : Infinite ι
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    J : Ideal ((i : ι) → R i) :=
      let __spread.0 := AddMonoidHom.mrange DFinsupp.coeFnAddMonoidHom;
      { toAddSubmonoid := __spread.0, smul_mem' := ⋯ }
    I : Ideal ((i : ι) → R i)
    max : I.IsMaximal
    le : LE.le J I
    ⊢ Exists fun I => Exists fun x => Not (Membership.mem (Set.range (PrimeSpectru …
  -/
  refine ⟨I, max, fun ⟨⟨i, p⟩, eq⟩ ↦ ?_⟩
  -- then I is not in the range of `sigmaToPi`
  have : ⇑(DFinsupp.single i 1) ∉ (sigmaToPi R ⟨i, p⟩).asIdeal := by
    simpa using p.1.ne_top_iff_one.mp p.2.ne_top
  /-
    ι : Type u_3
    R : ι → Type u_2
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : Infinite ι
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    J : Ideal ((i : ι) → R i) :=
      let __spread.0 := AddMonoidHom.mrange DFinsupp.coeFnAddMonoidHom;
      { toAddSubmonoid := __spread.0, smul_mem' := ⋯ }
    I : Ideal ((i : ι) → R i)
    max : I.IsMaximal
    le : LE.le J I
    x✝ : Membership.mem (Set.range (PrimeSpectrum.sigmaToPi R)) { asIdeal := I, is …
    i : ι
    p : PrimeSpectrum (R i)
    eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) { asIdeal := I, isPrime := ⋯ }
    this : Not (Membership.mem (PrimeSpectrum.sigmaToPi R ⟨i, p⟩).asIdeal ⇑(DFinsu …
    ⊢ False
  -/
  rw [eq] at this
  /-
    ι : Type u_3
    R : ι → Type u_2
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : Infinite ι
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    J : Ideal ((i : ι) → R i) :=
      let __spread.0 := AddMonoidHom.mrange DFinsupp.coeFnAddMonoidHom;
      { toAddSubmonoid := __spread.0, smul_mem' := ⋯ }
    I : Ideal ((i : ι) → R i)
    max : I.IsMaximal
    le : LE.le J I
    x✝ : Membership.mem (Set.range (PrimeSpectrum.sigmaToPi R)) { asIdeal := I, is …
    i : ι
    p : PrimeSpectrum (R i)
    eq : Eq (PrimeSpectrum.sigmaToPi R ⟨i, p⟩) { asIdeal := I, isPrime := ⋯ }
    this : Not (Membership.mem { asIdeal := I, isPrime := ⋯ }.asIdeal ⇑(DFinsupp.s …
    ⊢ False
  -/
  exact this (le ⟨.single i 1, rfl⟩)
  /-
    🎉 no goals
  -/


theorem sigmaToPi_not_surjective_of_infinite : ¬ (sigmaToPi R).Surjective := fun surj ↦
  have ⟨_, _, nmem⟩ := exists_maximal_nmem_range_sigmaToPi_of_infinite R
  (Set.range_eq_univ.mpr surj ▸ nmem) ⟨⟩


theorem image_specComap_zeroLocus_eq_zeroLocus_comap (hf : Surjective f) (I : Ideal S) :
    f.specComap '' zeroLocus I = zeroLocus (I.comap f) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal S
    ⊢ Eq (Set.image f.specComap (PrimeSpectrum.zeroLocus ↑I)) (PrimeSpectrum.zeroL …
  -/
  simp only [Set.ext_iff, Set.mem_image, mem_zeroLocus, SetLike.coe_subset_coe]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal S
    ⊢ ∀ (x : PrimeSpectrum R), Iff (Exists fun x_1 => And (LE.le I x_1.asIdeal) (E …
  -/
  refine fun p => ⟨?_, fun h_I_p => ?_⟩
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      p : PrimeSpectrum R
      ⊢ (Exists fun x => And (LE.le I x.asIdeal) (Eq (f.specComap x) p)) → LE.le (Id …
    -/
  · rintro ⟨p, hp, rfl⟩ a ha
    /-
      case refine_1.intro.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      p : PrimeSpectrum S
      hp : LE.le I p.asIdeal
      a : R
      ha : Membership.mem (Ideal.comap f I) a
      ⊢ Membership.mem (f.specComap p).asIdeal a
    -/
    exact hp ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      p : PrimeSpectrum R
      h_I_p : LE.le (Ideal.comap f I) p.asIdeal
      ⊢ Exists fun x => And (LE.le I x.asIdeal) (Eq (f.specComap x) p)
    -/
  · have hp : ker f ≤ p.asIdeal := (Ideal.comap_mono bot_le).trans h_I_p
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal S
      p : PrimeSpectrum R
      h_I_p : LE.le (Ideal.comap f I) p.asIdeal
      hp : LE.le (RingHom.ker f) p.asIdeal
      ⊢ Exists fun x => And (LE.le I x.asIdeal) (Eq (f.specComap x) p)
    -/
    refine ⟨⟨p.asIdeal.map f, Ideal.map_isPrime_of_surjective hf hp⟩, fun x hx => ?_, ?_⟩
      /-
        case refine_2.refine_1
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x : S
        hx : Membership.mem I x
        ⊢ Membership.mem { asIdeal := Ideal.map f p.asIdeal, isPrime := ⋯ }.asIdeal x
      -/
    · obtain ⟨x', rfl⟩ := hf x
      /-
        case refine_2.refine_1.intro
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x' : R
        hx : Membership.mem I (f x')
        ⊢ Membership.mem { asIdeal := Ideal.map f p.asIdeal, isPrime := ⋯ }.asIdeal (f …
      -/
      exact Ideal.mem_map_of_mem f (h_I_p hx)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        ⊢ Eq (f.specComap { asIdeal := Ideal.map f p.asIdeal, isPrime := ⋯ }) p
      -/
    · ext x
      /-
        case refine_2.refine_2.asIdeal.h
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x : R
        ⊢ Iff (Membership.mem (f.specComap { asIdeal := Ideal.map f p.asIdeal, isPrime …
      -/
      rw [specComap_asIdeal, Ideal.mem_comap, Ideal.mem_map_iff_of_surjective f hf]
      /-
        case refine_2.refine_2.asIdeal.h
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x : R
        ⊢ Iff (Exists fun x_1 => And (Membership.mem p.asIdeal x_1) (Eq (f x_1) (f x)) …
      -/
      refine ⟨?_, fun hx => ⟨x, hx, rfl⟩⟩
      /-
        case refine_2.refine_2.asIdeal.h
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x : R
        ⊢ (Exists fun x_1 => And (Membership.mem p.asIdeal x_1) (Eq (f x_1) (f x))) →  …
      -/
      rintro ⟨x', hx', heq⟩
      /-
        case refine_2.refine_2.asIdeal.h.intro.intro
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x x' : R
        hx' : Membership.mem p.asIdeal x'
        heq : Eq (f x') (f x)
        ⊢ Membership.mem p.asIdeal x
      -/
      rw [← sub_sub_cancel x' x]
      /-
        case refine_2.refine_2.asIdeal.h.intro.intro
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x x' : R
        hx' : Membership.mem p.asIdeal x'
        heq : Eq (f x') (f x)
        ⊢ Membership.mem p.asIdeal (HSub.hSub x' (HSub.hSub x' x))
      -/
      refine p.asIdeal.sub_mem hx' (hp ?_)
      /-
        case refine_2.refine_2.asIdeal.h.intro.intro
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        I : Ideal S
        p : PrimeSpectrum R
        h_I_p : LE.le (Ideal.comap f I) p.asIdeal
        hp : LE.le (RingHom.ker f) p.asIdeal
        x x' : R
        hx' : Membership.mem p.asIdeal x'
        heq : Eq (f x') (f x)
        ⊢ Membership.mem (RingHom.ker f) (HSub.hSub x' x)
      -/
      rwa [mem_ker, map_sub, sub_eq_zero]
      /-
        🎉 no goals
      -/


theorem range_specComap_of_surjective (hf : Surjective f) :
    Set.range f.specComap = zeroLocus (ker f) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ Eq (Set.range f.specComap) (PrimeSpectrum.zeroLocus ↑(RingHom.ker f))
  -/
  rw [← Set.image_univ]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ Eq (Set.image f.specComap Set.univ) (PrimeSpectrum.zeroLocus ↑(RingHom.ker f))
  -/
  convert image_specComap_zeroLocus_eq_zeroLocus_comap _ _ hf _
  /-
    case h.e'_2.h.e'_4
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ Eq Set.univ (PrimeSpectrum.zeroLocus ↑Bot.bot)
  -/
  rw [zeroLocus_bot]
  /-
    🎉 no goals
  -/


