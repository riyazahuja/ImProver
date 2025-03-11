/-- The projective spectrum of a graded commutative ring is the subtype of all homogeneous ideals
that are prime and do not contain the irrelevant ideal. -/
@[ext]
structure ProjectiveSpectrum where
  asHomogeneousIdeal : HomogeneousIdeal 𝒜
  isPrime : asHomogeneousIdeal.toIdeal.IsPrime
  not_irrelevant_le : ¬HomogeneousIdeal.irrelevant 𝒜 ≤ asHomogeneousIdeal


/-- The zero locus of a set `s` of elements of a commutative ring `A` is the set of all relevant
homogeneous prime ideals of the ring that contain the set `s`.

An element `f` of `A` can be thought of as a dependent function on the projective spectrum of `𝒜`.
At a point `x` (a homogeneous prime ideal) the function (i.e., element) `f` takes values in the
quotient ring `A` modulo the prime ideal `x`. In this manner, `zeroLocus s` is exactly the subset
of `ProjectiveSpectrum 𝒜` where all "functions" in `s` vanish simultaneously. -/
def zeroLocus (s : Set A) : Set (ProjectiveSpectrum 𝒜) :=
  { x | s ⊆ x.asHomogeneousIdeal }


@[simp]
theorem mem_zeroLocus (x : ProjectiveSpectrum 𝒜) (s : Set A) :
    x ∈ zeroLocus 𝒜 s ↔ s ⊆ x.asHomogeneousIdeal :=
  Iff.rfl


@[simp]
theorem zeroLocus_span (s : Set A) : zeroLocus 𝒜 (Ideal.span s) = zeroLocus 𝒜 s := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s : Set A
    ⊢ Eq (ProjectiveSpectrum.zeroLocus 𝒜 ↑(Ideal.span s)) (ProjectiveSpectrum.zero …
  -/
  ext x
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s : Set A
    x : ProjectiveSpectrum 𝒜
    ⊢ Iff (Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 ↑(Ideal.span s)) x) (Mem …
  -/
  exact (Submodule.gi _ _).gc s x.asHomogeneousIdeal.toIdeal
  /-
    🎉 no goals
  -/


/-- The vanishing ideal of a set `t` of points of the projective spectrum of a commutative ring `R`
is the intersection of all the relevant homogeneous prime ideals in the set `t`.

An element `f` of `A` can be thought of as a dependent function on the projective spectrum of `𝒜`.
At a point `x` (a homogeneous prime ideal) the function (i.e., element) `f` takes values in the
quotient ring `A` modulo the prime ideal `x`. In this manner, `vanishingIdeal t` is exactly the
ideal of `A` consisting of all "functions" that vanish on all of `t`. -/
def vanishingIdeal (t : Set (ProjectiveSpectrum 𝒜)) : HomogeneousIdeal 𝒜 :=
  ⨅ (x : ProjectiveSpectrum 𝒜) (_ : x ∈ t), x.asHomogeneousIdeal


theorem coe_vanishingIdeal (t : Set (ProjectiveSpectrum 𝒜)) :
    (vanishingIdeal t : Set A) =
      { f | ∀ x : ProjectiveSpectrum 𝒜, x ∈ t → f ∈ x.asHomogeneousIdeal } := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    ⊢ Eq (↑(ProjectiveSpectrum.vanishingIdeal t)) (setOf fun f => ∀ (x : Projectiv …
  -/
  ext f
  rw [vanishingIdeal, SetLike.mem_coe, ← HomogeneousIdeal.mem_iff, HomogeneousIdeal.toIdeal_iInf,
    Submodule.mem_iInf]
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    f : A
    ⊢ Iff (∀ (i : ProjectiveSpectrum 𝒜), Membership.mem (iInf fun x => i.asHomogen …
  -/
  refine forall_congr' fun x => ?_
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    f : A
    x : ProjectiveSpectrum 𝒜
    ⊢ Iff (Membership.mem (iInf fun x_1 => x.asHomogeneousIdeal).toIdeal f) (Membe …
  -/
  rw [HomogeneousIdeal.toIdeal_iInf, Submodule.mem_iInf, HomogeneousIdeal.mem_iff]
  /-
    🎉 no goals
  -/


theorem mem_vanishingIdeal (t : Set (ProjectiveSpectrum 𝒜)) (f : A) :
    f ∈ vanishingIdeal t ↔ ∀ x : ProjectiveSpectrum 𝒜, x ∈ t → f ∈ x.asHomogeneousIdeal := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    f : A
    ⊢ Iff (Membership.mem (ProjectiveSpectrum.vanishingIdeal t) f) (∀ (x : Project …
  -/
  rw [← SetLike.mem_coe, coe_vanishingIdeal, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem vanishingIdeal_singleton (x : ProjectiveSpectrum 𝒜) :
    vanishingIdeal ({x} : Set (ProjectiveSpectrum 𝒜)) = x.asHomogeneousIdeal := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x : ProjectiveSpectrum 𝒜
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (Singleton.singleton x)) x.asHomogeneo …
  -/
  simp [vanishingIdeal]
  /-
    🎉 no goals
  -/


theorem subset_zeroLocus_iff_le_vanishingIdeal (t : Set (ProjectiveSpectrum 𝒜)) (I : Ideal A) :
    t ⊆ zeroLocus 𝒜 I ↔ I ≤ (vanishingIdeal t).toIdeal :=
  ⟨fun h _ k => (mem_vanishingIdeal _ _).mpr fun _ j => (mem_zeroLocus _ _ _).mpr (h j) k, fun h =>
    fun x j =>
    (mem_zeroLocus _ _ _).mpr (le_trans h fun _ h => ((mem_vanishingIdeal _ _).mp h) x j)⟩


/-- `zeroLocus` and `vanishingIdeal` form a galois connection. -/
theorem gc_ideal :
    @GaloisConnection (Ideal A) (Set (ProjectiveSpectrum 𝒜))ᵒᵈ _ _
      (fun I => zeroLocus 𝒜 I) fun t => (vanishingIdeal t).toIdeal :=
  fun I t => subset_zeroLocus_iff_le_vanishingIdeal t I


/-- `zeroLocus` and `vanishingIdeal` form a galois connection. -/
theorem gc_set :
    @GaloisConnection (Set A) (Set (ProjectiveSpectrum 𝒜))ᵒᵈ _ _
      (fun s => zeroLocus 𝒜 s) fun t => vanishingIdeal t := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ⊢ GaloisConnection (fun s => ProjectiveSpectrum.zeroLocus 𝒜 s) fun t => ↑(Proj …
  -/
  have ideal_gc : GaloisConnection Ideal.span _ := (Submodule.gi A _).gc
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ideal_gc : GaloisConnection Ideal.span SetLike.coe
    ⊢ GaloisConnection (fun s => ProjectiveSpectrum.zeroLocus 𝒜 s) fun t => ↑(Proj …
  -/
  simpa [zeroLocus_span, Function.comp_def] using GaloisConnection.compose ideal_gc (gc_ideal 𝒜)
  /-
    🎉 no goals
  -/


theorem gc_homogeneousIdeal :
    @GaloisConnection (HomogeneousIdeal 𝒜) (Set (ProjectiveSpectrum 𝒜))ᵒᵈ _ _
      (fun I => zeroLocus 𝒜 I) fun t => vanishingIdeal t :=
  fun I t => by
  simpa [show I.toIdeal ≤ (vanishingIdeal t).toIdeal ↔ I ≤ vanishingIdeal t from Iff.rfl] using
    subset_zeroLocus_iff_le_vanishingIdeal t I.toIdeal


theorem subset_zeroLocus_iff_subset_vanishingIdeal (t : Set (ProjectiveSpectrum 𝒜)) (s : Set A) :
    t ⊆ zeroLocus 𝒜 s ↔ s ⊆ vanishingIdeal t :=
  (gc_set _) s t


theorem subset_vanishingIdeal_zeroLocus (s : Set A) : s ⊆ vanishingIdeal (zeroLocus 𝒜 s) :=
  (gc_set _).le_u_l s


theorem ideal_le_vanishingIdeal_zeroLocus (I : Ideal A) :
    I ≤ (vanishingIdeal (zeroLocus 𝒜 I)).toIdeal :=
  (gc_ideal _).le_u_l I


theorem homogeneousIdeal_le_vanishingIdeal_zeroLocus (I : HomogeneousIdeal 𝒜) :
    I ≤ vanishingIdeal (zeroLocus 𝒜 I) :=
  (gc_homogeneousIdeal _).le_u_l I


theorem subset_zeroLocus_vanishingIdeal (t : Set (ProjectiveSpectrum 𝒜)) :
    t ⊆ zeroLocus 𝒜 (vanishingIdeal t) :=
  (gc_ideal _).l_u_le t


theorem zeroLocus_anti_mono {s t : Set A} (h : s ⊆ t) : zeroLocus 𝒜 t ⊆ zeroLocus 𝒜 s :=
  (gc_set _).monotone_l h


theorem zeroLocus_anti_mono_ideal {s t : Ideal A} (h : s ≤ t) :
    zeroLocus 𝒜 (t : Set A) ⊆ zeroLocus 𝒜 (s : Set A) :=
  (gc_ideal _).monotone_l h


theorem zeroLocus_anti_mono_homogeneousIdeal {s t : HomogeneousIdeal 𝒜} (h : s ≤ t) :
    zeroLocus 𝒜 (t : Set A) ⊆ zeroLocus 𝒜 (s : Set A) :=
  (gc_homogeneousIdeal _).monotone_l h


theorem vanishingIdeal_anti_mono {s t : Set (ProjectiveSpectrum 𝒜)} (h : s ⊆ t) :
    vanishingIdeal t ≤ vanishingIdeal s :=
  (gc_ideal _).monotone_u h


theorem zeroLocus_bot : zeroLocus 𝒜 ((⊥ : Ideal A) : Set A) = Set.univ :=
  (gc_ideal 𝒜).l_bot


@[simp]
theorem zeroLocus_singleton_zero : zeroLocus 𝒜 ({0} : Set A) = Set.univ :=
  zeroLocus_bot _


@[simp]
theorem zeroLocus_empty : zeroLocus 𝒜 (∅ : Set A) = Set.univ :=
  (gc_set 𝒜).l_bot


@[simp]
theorem vanishingIdeal_univ : vanishingIdeal (∅ : Set (ProjectiveSpectrum 𝒜)) = ⊤ := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal EmptyCollection.emptyCollection) Top.top
  -/
  simpa using (gc_ideal _).u_top
  /-
    🎉 no goals
  -/


theorem zeroLocus_empty_of_one_mem {s : Set A} (h : (1 : A) ∈ s) : zeroLocus 𝒜 s = ∅ :=
  Set.eq_empty_iff_forall_not_mem.mpr fun x hx =>
    (inferInstance : x.asHomogeneousIdeal.toIdeal.IsPrime).ne_top <|
      x.asHomogeneousIdeal.toIdeal.eq_top_iff_one.mpr <| hx h


@[simp]
theorem zeroLocus_singleton_one : zeroLocus 𝒜 ({1} : Set A) = ∅ :=
  zeroLocus_empty_of_one_mem 𝒜 (Set.mem_singleton (1 : A))


@[simp]
theorem zeroLocus_univ : zeroLocus 𝒜 (Set.univ : Set A) = ∅ :=
  zeroLocus_empty_of_one_mem _ (Set.mem_univ 1)


theorem zeroLocus_sup_ideal (I J : Ideal A) :
    zeroLocus 𝒜 ((I ⊔ J : Ideal A) : Set A) = zeroLocus _ I ∩ zeroLocus _ J :=
  (gc_ideal 𝒜).l_sup


theorem zeroLocus_sup_homogeneousIdeal (I J : HomogeneousIdeal 𝒜) :
    zeroLocus 𝒜 ((I ⊔ J : HomogeneousIdeal 𝒜) : Set A) = zeroLocus _ I ∩ zeroLocus _ J :=
  (gc_homogeneousIdeal 𝒜).l_sup


theorem zeroLocus_union (s s' : Set A) : zeroLocus 𝒜 (s ∪ s') = zeroLocus _ s ∩ zeroLocus _ s' :=
  (gc_set 𝒜).l_sup


theorem vanishingIdeal_union (t t' : Set (ProjectiveSpectrum 𝒜)) :
    vanishingIdeal (t ∪ t') = vanishingIdeal t ⊓ vanishingIdeal t' := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t t' : Set (ProjectiveSpectrum 𝒜)
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (Union.union t t')) (Min.min (Projecti …
  -/
  ext1; exact (gc_ideal 𝒜).u_inf
        /-
          🎉 no goals
        -/


theorem zeroLocus_iSup_ideal {γ : Sort*} (I : γ → Ideal A) :
    zeroLocus _ ((⨆ i, I i : Ideal A) : Set A) = ⋂ i, zeroLocus 𝒜 (I i) :=
  (gc_ideal 𝒜).l_iSup


theorem zeroLocus_iSup_homogeneousIdeal {γ : Sort*} (I : γ → HomogeneousIdeal 𝒜) :
    zeroLocus _ ((⨆ i, I i : HomogeneousIdeal 𝒜) : Set A) = ⋂ i, zeroLocus 𝒜 (I i) :=
  (gc_homogeneousIdeal 𝒜).l_iSup


theorem zeroLocus_iUnion {γ : Sort*} (s : γ → Set A) :
    zeroLocus 𝒜 (⋃ i, s i) = ⋂ i, zeroLocus 𝒜 (s i) :=
  (gc_set 𝒜).l_iSup


theorem zeroLocus_bUnion (s : Set (Set A)) :
    zeroLocus 𝒜 (⋃ s' ∈ s, s' : Set A) = ⋂ s' ∈ s, zeroLocus 𝒜 s' := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s : Set (Set A)
    ⊢ Eq (ProjectiveSpectrum.zeroLocus 𝒜 (Set.iUnion fun s' => Set.iUnion fun h => …
  -/
  simp only [zeroLocus_iUnion]
  /-
    🎉 no goals
  -/


theorem vanishingIdeal_iUnion {γ : Sort*} (t : γ → Set (ProjectiveSpectrum 𝒜)) :
    vanishingIdeal (⋃ i, t i) = ⨅ i, vanishingIdeal (t i) :=
  HomogeneousIdeal.toIdeal_injective <| by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      γ : Sort u_3
      t : γ → Set (ProjectiveSpectrum 𝒜)
      ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (Set.iUnion fun i => t i)).toIdeal (iI …
    -/
    convert (gc_ideal 𝒜).u_iInf; exact HomogeneousIdeal.toIdeal_iInf _
                                 /-
                                   🎉 no goals
                                 -/


theorem zeroLocus_inf (I J : Ideal A) :
    zeroLocus 𝒜 ((I ⊓ J : Ideal A) : Set A) = zeroLocus 𝒜 I ∪ zeroLocus 𝒜 J :=
  Set.ext fun x => x.isPrime.inf_le


theorem union_zeroLocus (s s' : Set A) :
    zeroLocus 𝒜 s ∪ zeroLocus 𝒜 s' = zeroLocus 𝒜 (Ideal.span s ⊓ Ideal.span s' : Ideal A) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s s' : Set A
    ⊢ Eq (Union.union (ProjectiveSpectrum.zeroLocus 𝒜 s) (ProjectiveSpectrum.zeroL …
  -/
  rw [zeroLocus_inf]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s s' : Set A
    ⊢ Eq (Union.union (ProjectiveSpectrum.zeroLocus 𝒜 s) (ProjectiveSpectrum.zeroL …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem zeroLocus_mul_ideal (I J : Ideal A) :
    zeroLocus 𝒜 ((I * J : Ideal A) : Set A) = zeroLocus 𝒜 I ∪ zeroLocus 𝒜 J :=
  Set.ext fun x => x.isPrime.mul_le


theorem zeroLocus_mul_homogeneousIdeal (I J : HomogeneousIdeal 𝒜) :
    zeroLocus 𝒜 ((I * J : HomogeneousIdeal 𝒜) : Set A) = zeroLocus 𝒜 I ∪ zeroLocus 𝒜 J :=
  Set.ext fun x => x.isPrime.mul_le


theorem zeroLocus_singleton_mul (f g : A) :
    zeroLocus 𝒜 ({f * g} : Set A) = zeroLocus 𝒜 {f} ∪ zeroLocus 𝒜 {g} :=
                      /-
                        R : Type u_1
                        A : Type u_2
                        inst✝³ : CommSemiring R
                        inst✝² : CommRing A
                        inst✝¹ : Algebra R A
                        𝒜 : Nat → Submodule R A
                        inst✝ : GradedAlgebra 𝒜
                        f g : A
                        x : ProjectiveSpectrum 𝒜
                        ⊢ Iff (Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 (Singleton.singleton (HM …
                      -/
  Set.ext fun x => by simpa using x.isPrime.mul_mem_iff_mem_or_mem
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem zeroLocus_singleton_pow (f : A) (n : ℕ) (hn : 0 < n) :
    zeroLocus 𝒜 ({f ^ n} : Set A) = zeroLocus 𝒜 {f} :=
                      /-
                        R : Type u_1
                        A : Type u_2
                        inst✝³ : CommSemiring R
                        inst✝² : CommRing A
                        inst✝¹ : Algebra R A
                        𝒜 : Nat → Submodule R A
                        inst✝ : GradedAlgebra 𝒜
                        f : A
                        n : Nat
                        hn : LT.lt 0 n
                        x : ProjectiveSpectrum 𝒜
                        ⊢ Iff (Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 (Singleton.singleton (HP …
                      -/
  Set.ext fun x => by simpa using x.isPrime.pow_mem_iff_mem n hn
                      /-
                        🎉 no goals
                      -/


theorem sup_vanishingIdeal_le (t t' : Set (ProjectiveSpectrum 𝒜)) :
    vanishingIdeal t ⊔ vanishingIdeal t' ≤ vanishingIdeal (t ∩ t') := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t t' : Set (ProjectiveSpectrum 𝒜)
    ⊢ LE.le (Max.max (ProjectiveSpectrum.vanishingIdeal t) (ProjectiveSpectrum.van …
  -/
  intro r
  rw [← HomogeneousIdeal.mem_iff, HomogeneousIdeal.toIdeal_sup, mem_vanishingIdeal,
    Submodule.mem_sup]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t t' : Set (ProjectiveSpectrum 𝒜)
    r : A
    ⊢ (Exists fun y => And (Membership.mem (ProjectiveSpectrum.vanishingIdeal t).t …
  -/
  rintro ⟨f, hf, g, hg, rfl⟩ x ⟨hxt, hxt'⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t t' : Set (ProjectiveSpectrum 𝒜)
    f : A
    hf : Membership.mem (ProjectiveSpectrum.vanishingIdeal t).toIdeal f
    g : A
    hg : Membership.mem (ProjectiveSpectrum.vanishingIdeal t').toIdeal g
    x : ProjectiveSpectrum 𝒜
    hxt : Membership.mem t x
    hxt' : Membership.mem t' x
    ⊢ Membership.mem x.asHomogeneousIdeal (HAdd.hAdd f g)
  -/
  erw [mem_vanishingIdeal] at hf hg
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t t' : Set (ProjectiveSpectrum 𝒜)
    f : A
    hf : ∀ (x : ProjectiveSpectrum 𝒜), Membership.mem t x → Membership.mem x.asHom …
    g : A
    hg : ∀ (x : ProjectiveSpectrum 𝒜), Membership.mem t' x → Membership.mem x.asHo …
    x : ProjectiveSpectrum 𝒜
    hxt : Membership.mem t x
    hxt' : Membership.mem t' x
    ⊢ Membership.mem x.asHomogeneousIdeal (HAdd.hAdd f g)
  -/
                              /-
                                🎉 no goals
                              -/
  apply Submodule.add_mem <;> solve_by_elim
                              /-
                                🎉 no goals
                              -/


theorem mem_compl_zeroLocus_iff_not_mem {f : A} {I : ProjectiveSpectrum 𝒜} :
    I ∈ (zeroLocus 𝒜 {f} : Set (ProjectiveSpectrum 𝒜))ᶜ ↔ f ∉ I.asHomogeneousIdeal := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    I : ProjectiveSpectrum 𝒜
    ⊢ Iff (Membership.mem (HasCompl.compl (ProjectiveSpectrum.zeroLocus 𝒜 (Singlet …
  -/
  rw [Set.mem_compl_iff, mem_zeroLocus, Set.singleton_subset_iff]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The Zariski topology on the prime spectrum of a commutative ring is defined via the closed sets
of the topology: they are exactly those sets that are the zero locus of a subset of the ring. -/
instance zariskiTopology : TopologicalSpace (ProjectiveSpectrum 𝒜) :=
                                                                                       /-
                                                                                         R : Type u_1
                                                                                         A : Type u_2
                                                                                         inst✝³ : CommSemiring R
                                                                                         inst✝² : CommRing A
                                                                                         inst✝¹ : Algebra R A
                                                                                         𝒜 : Nat → Submodule R A
                                                                                         inst✝ : GradedAlgebra 𝒜
                                                                                         ⊢ Eq (ProjectiveSpectrum.zeroLocus 𝒜 Set.univ) EmptyCollection.emptyCollection
                                                                                       -/
  TopologicalSpace.ofClosed (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) ⟨Set.univ, by simp⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
    (by
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        ⊢ ∀ (A_1 : Set (Set (ProjectiveSpectrum 𝒜))), HasSubset.Subset A_1 (Set.range  …
      -/
      intro Zs h
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        Zs : Set (Set (ProjectiveSpectrum 𝒜))
        h : HasSubset.Subset Zs (Set.range (ProjectiveSpectrum.zeroLocus 𝒜))
        ⊢ Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) Zs.sInter
      -/
      rw [Set.sInter_eq_iInter]
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        Zs : Set (Set (ProjectiveSpectrum 𝒜))
        h : HasSubset.Subset Zs (Set.range (ProjectiveSpectrum.zeroLocus 𝒜))
        ⊢ Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) (Set.iInter fun  …
      -/
      let f : Zs → Set _ := fun i => Classical.choose (h i.2)
      have H : (Set.iInter fun i ↦ zeroLocus 𝒜 (f i)) ∈ Set.range (zeroLocus 𝒜) :=
        ⟨_, zeroLocus_iUnion 𝒜 _⟩
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        Zs : Set (Set (ProjectiveSpectrum 𝒜))
        h : HasSubset.Subset Zs (Set.range (ProjectiveSpectrum.zeroLocus 𝒜))
        f : ↑Zs → Set A := fun i => Classical.choose ⋯
        H : Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) (Set.iInter fu …
        ⊢ Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) (Set.iInter fun  …
      -/
      convert H using 2
      /-
        case h.e'_5.h.e'_3
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        Zs : Set (Set (ProjectiveSpectrum 𝒜))
        h : HasSubset.Subset Zs (Set.range (ProjectiveSpectrum.zeroLocus 𝒜))
        f : ↑Zs → Set A := fun i => Classical.choose ⋯
        H : Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) (Set.iInter fu …
        ⊢ Eq (fun i => ↑i) fun i => ProjectiveSpectrum.zeroLocus 𝒜 (f i)
      -/
      funext i
      /-
        case h.e'_5.h.e'_3.h
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        Zs : Set (Set (ProjectiveSpectrum 𝒜))
        h : HasSubset.Subset Zs (Set.range (ProjectiveSpectrum.zeroLocus 𝒜))
        f : ↑Zs → Set A := fun i => Classical.choose ⋯
        H : Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) (Set.iInter fu …
        i : ↑Zs
        ⊢ Eq (↑i) (ProjectiveSpectrum.zeroLocus 𝒜 (f i))
      -/
      exact (Classical.choose_spec (h i.2)).symm)
      /-
        🎉 no goals
      -/
    (by
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        ⊢ ∀ (A_1 : Set (ProjectiveSpectrum 𝒜)), Membership.mem (Set.range (ProjectiveS …
      -/
      rintro _ ⟨s, rfl⟩ _ ⟨t, rfl⟩
      /-
        case intro.intro
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        s t : Set A
        ⊢ Membership.mem (Set.range (ProjectiveSpectrum.zeroLocus 𝒜)) (Union.union (Pr …
      -/
      exact ⟨_, (union_zeroLocus 𝒜 s t).symm⟩)
      /-
        🎉 no goals
      -/


/-- The underlying topology of `Proj` is the projective spectrum of graded ring `A`. -/
def top : TopCat :=
  TopCat.of (ProjectiveSpectrum 𝒜)


theorem isOpen_iff (U : Set (ProjectiveSpectrum 𝒜)) : IsOpen U ↔ ∃ s, Uᶜ = zeroLocus 𝒜 s := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    U : Set (ProjectiveSpectrum 𝒜)
    ⊢ Iff (IsOpen U) (Exists fun s => Eq (HasCompl.compl U) (ProjectiveSpectrum.ze …
  -/
  simp only [@eq_comm _ Uᶜ]; rfl
                             /-
                               🎉 no goals
                             -/


theorem isClosed_iff_zeroLocus (Z : Set (ProjectiveSpectrum 𝒜)) :
                                              /-
                                                R : Type u_1
                                                A : Type u_2
                                                inst✝³ : CommSemiring R
                                                inst✝² : CommRing A
                                                inst✝¹ : Algebra R A
                                                𝒜 : Nat → Submodule R A
                                                inst✝ : GradedAlgebra 𝒜
                                                Z : Set (ProjectiveSpectrum 𝒜)
                                                ⊢ Iff (IsClosed Z) (Exists fun s => Eq Z (ProjectiveSpectrum.zeroLocus 𝒜 s))
                                              -/
    IsClosed Z ↔ ∃ s, Z = zeroLocus 𝒜 s := by rw [← isOpen_compl_iff, isOpen_iff, compl_compl]
                                              /-
                                                🎉 no goals
                                              -/


theorem isClosed_zeroLocus (s : Set A) : IsClosed (zeroLocus 𝒜 s) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s : Set A
    ⊢ IsClosed (ProjectiveSpectrum.zeroLocus 𝒜 s)
  -/
  rw [isClosed_iff_zeroLocus]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    s : Set A
    ⊢ Exists fun s_1 => Eq (ProjectiveSpectrum.zeroLocus 𝒜 s) (ProjectiveSpectrum. …
  -/
  exact ⟨s, rfl⟩
  /-
    🎉 no goals
  -/


theorem zeroLocus_vanishingIdeal_eq_closure (t : Set (ProjectiveSpectrum 𝒜)) :
    zeroLocus 𝒜 (vanishingIdeal t : Set A) = closure t := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    ⊢ Eq (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpectrum.vanishingIdeal t)) ( …
  -/
  apply Set.Subset.antisymm
    /-
      case h₁
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      t : Set (ProjectiveSpectrum 𝒜)
      ⊢ HasSubset.Subset (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpectrum.vanish …
    -/
  · rintro x hx t' ⟨ht', ht⟩
    /-
      case h₁.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      t : Set (ProjectiveSpectrum 𝒜)
      x : ProjectiveSpectrum 𝒜
      hx : Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpectrum.vanis …
      t' : Set (ProjectiveSpectrum 𝒜)
      ht' : IsClosed t'
      ht : HasSubset.Subset t t'
      ⊢ Membership.mem t' x
    -/
    obtain ⟨fs, rfl⟩ : ∃ s, t' = zeroLocus 𝒜 s := by rwa [isClosed_iff_zeroLocus] at ht'
    /-
      case h₁.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      t : Set (ProjectiveSpectrum 𝒜)
      x : ProjectiveSpectrum 𝒜
      hx : Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpectrum.vanis …
      fs : Set A
      ht' : IsClosed (ProjectiveSpectrum.zeroLocus 𝒜 fs)
      ht : HasSubset.Subset t (ProjectiveSpectrum.zeroLocus 𝒜 fs)
      ⊢ Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 fs) x
    -/
    rw [subset_zeroLocus_iff_subset_vanishingIdeal] at ht
    /-
      case h₁.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      t : Set (ProjectiveSpectrum 𝒜)
      x : ProjectiveSpectrum 𝒜
      hx : Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpectrum.vanis …
      fs : Set A
      ht' : IsClosed (ProjectiveSpectrum.zeroLocus 𝒜 fs)
      ht : HasSubset.Subset fs ↑(ProjectiveSpectrum.vanishingIdeal t)
      ⊢ Membership.mem (ProjectiveSpectrum.zeroLocus 𝒜 fs) x
    -/
    exact Set.Subset.trans ht hx
    /-
      🎉 no goals
    -/
    /-
      case h₂
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      t : Set (ProjectiveSpectrum 𝒜)
      ⊢ HasSubset.Subset (closure t) (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpe …
    -/
  · rw [(isClosed_zeroLocus _ _).closure_subset_iff]
    /-
      case h₂
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      t : Set (ProjectiveSpectrum 𝒜)
      ⊢ HasSubset.Subset t (ProjectiveSpectrum.zeroLocus 𝒜 ↑(ProjectiveSpectrum.vani …
    -/
    exact subset_zeroLocus_vanishingIdeal 𝒜 t
    /-
      🎉 no goals
    -/


theorem vanishingIdeal_closure (t : Set (ProjectiveSpectrum 𝒜)) :
    vanishingIdeal (closure t) = vanishingIdeal t := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (closure t)) (ProjectiveSpectrum.vanis …
  -/
  have := (gc_ideal 𝒜).u_l_u_eq_u t
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    this : Eq (ProjectiveSpectrum.vanishingIdeal (ProjectiveSpectrum.zeroLocus 𝒜 ↑ …
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (closure t)) (ProjectiveSpectrum.vanis …
  -/
  ext1
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    this : Eq (ProjectiveSpectrum.vanishingIdeal (ProjectiveSpectrum.zeroLocus 𝒜 ↑ …
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (closure t)).toIdeal (ProjectiveSpectr …
  -/
  erw [zeroLocus_vanishingIdeal_eq_closure 𝒜 t] at this
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    t : Set (ProjectiveSpectrum 𝒜)
    this : Eq (ProjectiveSpectrum.vanishingIdeal (closure t)).toIdeal (ProjectiveS …
    ⊢ Eq (ProjectiveSpectrum.vanishingIdeal (closure t)).toIdeal (ProjectiveSpectr …
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- `basicOpen r` is the open subset containing all prime ideals not containing `r`. -/
def basicOpen (r : A) : TopologicalSpace.Opens (ProjectiveSpectrum 𝒜) where
  carrier := { x | r ∉ x.asHomogeneousIdeal }
  is_open' := ⟨{r}, Set.ext fun _ => Set.singleton_subset_iff.trans <| Classical.not_not.symm⟩


@[simp]
theorem mem_basicOpen (f : A) (x : ProjectiveSpectrum 𝒜) :
    x ∈ basicOpen 𝒜 f ↔ f ∉ x.asHomogeneousIdeal :=
  Iff.rfl


theorem mem_coe_basicOpen (f : A) (x : ProjectiveSpectrum 𝒜) :
    x ∈ (↑(basicOpen 𝒜 f) : Set (ProjectiveSpectrum 𝒜)) ↔ f ∉ x.asHomogeneousIdeal :=
  Iff.rfl


theorem isOpen_basicOpen {a : A} : IsOpen (basicOpen 𝒜 a : Set (ProjectiveSpectrum 𝒜)) :=
  (basicOpen 𝒜 a).isOpen


@[simp]
theorem basicOpen_eq_zeroLocus_compl (r : A) :
    (basicOpen 𝒜 r : Set (ProjectiveSpectrum 𝒜)) = (zeroLocus 𝒜 {r})ᶜ :=
                      /-
                        R : Type u_1
                        A : Type u_2
                        inst✝³ : CommSemiring R
                        inst✝² : CommRing A
                        inst✝¹ : Algebra R A
                        𝒜 : Nat → Submodule R A
                        inst✝ : GradedAlgebra 𝒜
                        r : A
                        x : ProjectiveSpectrum 𝒜
                        ⊢ Iff (Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 r)) x) (Membership.mem …
                      -/
  Set.ext fun x => by simp only [Set.mem_compl_iff, mem_zeroLocus, Set.singleton_subset_iff]; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[simp]
theorem basicOpen_one : basicOpen 𝒜 (1 : A) = ⊤ :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     inst✝³ : CommSemiring R
                                     inst✝² : CommRing A
                                     inst✝¹ : Algebra R A
                                     𝒜 : Nat → Submodule R A
                                     inst✝ : GradedAlgebra 𝒜
                                     ⊢ Eq ↑(ProjectiveSpectrum.basicOpen 𝒜 1) ↑Top.top
                                   -/
  TopologicalSpace.Opens.ext <| by simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem basicOpen_zero : basicOpen 𝒜 (0 : A) = ⊥ :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     inst✝³ : CommSemiring R
                                     inst✝² : CommRing A
                                     inst✝¹ : Algebra R A
                                     𝒜 : Nat → Submodule R A
                                     inst✝ : GradedAlgebra 𝒜
                                     ⊢ Eq ↑(ProjectiveSpectrum.basicOpen 𝒜 0) ↑Bot.bot
                                   -/
  TopologicalSpace.Opens.ext <| by simp
                                   /-
                                     🎉 no goals
                                   -/


theorem basicOpen_mul (f g : A) : basicOpen 𝒜 (f * g) = basicOpen 𝒜 f ⊓ basicOpen 𝒜 g :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     inst✝³ : CommSemiring R
                                     inst✝² : CommRing A
                                     inst✝¹ : Algebra R A
                                     𝒜 : Nat → Submodule R A
                                     inst✝ : GradedAlgebra 𝒜
                                     f g : A
                                     ⊢ Eq ↑(ProjectiveSpectrum.basicOpen 𝒜 (HMul.hMul f g)) ↑(Min.min (ProjectiveSp …
                                   -/
  TopologicalSpace.Opens.ext <| by simp [zeroLocus_singleton_mul]
                                   /-
                                     🎉 no goals
                                   -/


theorem basicOpen_mul_le_left (f g : A) : basicOpen 𝒜 (f * g) ≤ basicOpen 𝒜 f := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g : A
    ⊢ LE.le (ProjectiveSpectrum.basicOpen 𝒜 (HMul.hMul f g)) (ProjectiveSpectrum.b …
  -/
  rw [basicOpen_mul 𝒜 f g]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g : A
    ⊢ LE.le (Min.min (ProjectiveSpectrum.basicOpen 𝒜 f) (ProjectiveSpectrum.basicO …
  -/
  exact inf_le_left
  /-
    🎉 no goals
  -/


theorem basicOpen_mul_le_right (f g : A) : basicOpen 𝒜 (f * g) ≤ basicOpen 𝒜 g := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g : A
    ⊢ LE.le (ProjectiveSpectrum.basicOpen 𝒜 (HMul.hMul f g)) (ProjectiveSpectrum.b …
  -/
  rw [basicOpen_mul 𝒜 f g]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f g : A
    ⊢ LE.le (Min.min (ProjectiveSpectrum.basicOpen 𝒜 f) (ProjectiveSpectrum.basicO …
  -/
  exact inf_le_right
  /-
    🎉 no goals
  -/


@[simp]
theorem basicOpen_pow (f : A) (n : ℕ) (hn : 0 < n) : basicOpen 𝒜 (f ^ n) = basicOpen 𝒜 f :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     inst✝³ : CommSemiring R
                                     inst✝² : CommRing A
                                     inst✝¹ : Algebra R A
                                     𝒜 : Nat → Submodule R A
                                     inst✝ : GradedAlgebra 𝒜
                                     f : A
                                     n : Nat
                                     hn : LT.lt 0 n
                                     ⊢ Eq ↑(ProjectiveSpectrum.basicOpen 𝒜 (HPow.hPow f n)) ↑(ProjectiveSpectrum.ba …
                                   -/
  TopologicalSpace.Opens.ext <| by simpa using zeroLocus_singleton_pow 𝒜 f n hn
                                   /-
                                     🎉 no goals
                                   -/


theorem basicOpen_eq_union_of_projection (f : A) :
    basicOpen 𝒜 f = ⨆ i : ℕ, basicOpen 𝒜 (GradedAlgebra.proj 𝒜 i f) :=
  TopologicalSpace.Opens.ext <|
    Set.ext fun z => by
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        z : ProjectiveSpectrum 𝒜
        ⊢ Iff (Membership.mem (↑(ProjectiveSpectrum.basicOpen 𝒜 f)) z) (Membership.mem …
      -/
      erw [mem_coe_basicOpen, TopologicalSpace.Opens.mem_sSup]
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        z : ProjectiveSpectrum 𝒜
        ⊢ Iff (Not (Membership.mem z.asHomogeneousIdeal f)) (Exists fun u => And (Memb …
      -/
      constructor <;> intro hz
      · rcases show ∃ i, GradedAlgebra.proj 𝒜 i f ∉ z.asHomogeneousIdeal by
          contrapose! hz with H
          classical
          rw [← DirectSum.sum_support_decompose 𝒜 f]
          apply Ideal.sum_mem _ fun i _ => H i with ⟨i, hi⟩
        /-
          case mp.intro
          R : Type u_1
          A : Type u_2
          inst✝³ : CommSemiring R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          z : ProjectiveSpectrum 𝒜
          hz : Not (Membership.mem z.asHomogeneousIdeal f)
          i : Nat
          hi : Not (Membership.mem z.asHomogeneousIdeal ((GradedAlgebra.proj 𝒜 i) f))
          ⊢ Exists fun u => And (Membership.mem (Set.range fun i => ProjectiveSpectrum.b …
        -/
        exact ⟨basicOpen 𝒜 (GradedAlgebra.proj 𝒜 i f), ⟨i, rfl⟩, by rwa [mem_basicOpen]⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr
          R : Type u_1
          A : Type u_2
          inst✝³ : CommSemiring R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          z : ProjectiveSpectrum 𝒜
          hz : Exists fun u => And (Membership.mem (Set.range fun i => ProjectiveSpectru …
          ⊢ Not (Membership.mem z.asHomogeneousIdeal f)
        -/
      · obtain ⟨_, ⟨i, rfl⟩, hz⟩ := hz
        /-
          case mpr.intro.intro.intro
          R : Type u_1
          A : Type u_2
          inst✝³ : CommSemiring R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          z : ProjectiveSpectrum 𝒜
          i : Nat
          hz : Membership.mem ((fun i => ProjectiveSpectrum.basicOpen 𝒜 ((GradedAlgebra. …
          ⊢ Not (Membership.mem z.asHomogeneousIdeal f)
        -/
        exact fun rid => hz (z.1.2 i rid)
        /-
          🎉 no goals
        -/


theorem isTopologicalBasis_basic_opens :
    TopologicalSpace.IsTopologicalBasis
      (Set.range fun r : A => (basicOpen 𝒜 r : Set (ProjectiveSpectrum 𝒜))) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ⊢ TopologicalSpace.IsTopologicalBasis (Set.range fun r => ↑(ProjectiveSpectrum …
  -/
  apply TopologicalSpace.isTopologicalBasis_of_isOpen_of_nhds
    /-
      case h_open
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      ⊢ ∀ (u : Set (ProjectiveSpectrum 𝒜)), Membership.mem (Set.range fun r => ↑(Pro …
    -/
  · rintro _ ⟨r, rfl⟩
    /-
      case h_open.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      r : A
      ⊢ IsOpen ((fun r => ↑(ProjectiveSpectrum.basicOpen 𝒜 r)) r)
    -/
    exact isOpen_basicOpen 𝒜
    /-
      🎉 no goals
    -/
    /-
      case h_nhds
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      ⊢ ∀ (a : ProjectiveSpectrum 𝒜) (u : Set (ProjectiveSpectrum 𝒜)), Membership.me …
    -/
  · rintro p U hp ⟨s, hs⟩
    /-
      case h_nhds.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      p : ProjectiveSpectrum 𝒜
      U : Set (ProjectiveSpectrum 𝒜)
      hp : Membership.mem U p
      s : Set A
      hs : Eq (ProjectiveSpectrum.zeroLocus 𝒜 s) (HasCompl.compl U)
      ⊢ Exists fun v => And (Membership.mem (Set.range fun r => ↑(ProjectiveSpectrum …
    -/
    rw [← compl_compl U, Set.mem_compl_iff, ← hs, mem_zeroLocus, Set.not_subset] at hp
    /-
      case h_nhds.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      p : ProjectiveSpectrum 𝒜
      U : Set (ProjectiveSpectrum 𝒜)
      s : Set A
      hp : Exists fun a => And (Membership.mem s a) (Not (Membership.mem (↑p.asHomog …
      hs : Eq (ProjectiveSpectrum.zeroLocus 𝒜 s) (HasCompl.compl U)
      ⊢ Exists fun v => And (Membership.mem (Set.range fun r => ↑(ProjectiveSpectrum …
    -/
    obtain ⟨f, hfs, hfp⟩ := hp
    /-
      case h_nhds.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      p : ProjectiveSpectrum 𝒜
      U : Set (ProjectiveSpectrum 𝒜)
      s : Set A
      hs : Eq (ProjectiveSpectrum.zeroLocus 𝒜 s) (HasCompl.compl U)
      f : A
      hfs : Membership.mem s f
      hfp : Not (Membership.mem (↑p.asHomogeneousIdeal) f)
      ⊢ Exists fun v => And (Membership.mem (Set.range fun r => ↑(ProjectiveSpectrum …
    -/
    refine ⟨basicOpen 𝒜 f, ⟨f, rfl⟩, hfp, ?_⟩
    /-
      case h_nhds.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      p : ProjectiveSpectrum 𝒜
      U : Set (ProjectiveSpectrum 𝒜)
      s : Set A
      hs : Eq (ProjectiveSpectrum.zeroLocus 𝒜 s) (HasCompl.compl U)
      f : A
      hfs : Membership.mem s f
      hfp : Not (Membership.mem (↑p.asHomogeneousIdeal) f)
      ⊢ HasSubset.Subset (↑(ProjectiveSpectrum.basicOpen 𝒜 f)) U
    -/
    rw [← Set.compl_subset_compl, ← hs, basicOpen_eq_zeroLocus_compl, compl_compl]
    /-
      case h_nhds.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      p : ProjectiveSpectrum 𝒜
      U : Set (ProjectiveSpectrum 𝒜)
      s : Set A
      hs : Eq (ProjectiveSpectrum.zeroLocus 𝒜 s) (HasCompl.compl U)
      f : A
      hfs : Membership.mem s f
      hfp : Not (Membership.mem (↑p.asHomogeneousIdeal) f)
      ⊢ HasSubset.Subset (ProjectiveSpectrum.zeroLocus 𝒜 s) (ProjectiveSpectrum.zero …
    -/
    exact zeroLocus_anti_mono 𝒜 (Set.singleton_subset_iff.mpr hfs)
    /-
      🎉 no goals
    -/


instance : PartialOrder (ProjectiveSpectrum 𝒜) :=
                                                                     /-
                                                                       R : Type u_1
                                                                       A : Type u_2
                                                                       inst✝³ : CommSemiring R
                                                                       inst✝² : CommRing A
                                                                       inst✝¹ : Algebra R A
                                                                       𝒜 : Nat → Submodule R A
                                                                       inst✝ : GradedAlgebra 𝒜
                                                                       x✝¹ x✝ : ProjectiveSpectrum 𝒜
                                                                       asHomogeneousIdeal✝¹ : HomogeneousIdeal 𝒜
                                                                       isPrime✝¹ : asHomogeneousIdeal✝¹.toIdeal.IsPrime
                                                                       not_irrelevant_le✝¹ : Not (LE.le (HomogeneousIdeal.irrelevant 𝒜) asHomogeneous …
                                                                       asHomogeneousIdeal✝ : HomogeneousIdeal 𝒜
                                                                       isPrime✝ : asHomogeneousIdeal✝.toIdeal.IsPrime
                                                                       not_irrelevant_le✝ : Not (LE.le (HomogeneousIdeal.irrelevant 𝒜) asHomogeneousI …
                                                                       ⊢ Eq { asHomogeneousIdeal := asHomogeneousIdeal✝¹, isPrime := isPrime✝¹, not_i …
                                                                     -/
  PartialOrder.lift asHomogeneousIdeal fun ⟨_, _, _⟩ ⟨_, _, _⟩ => by simp only [mk.injEq, imp_self]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem as_ideal_le_as_ideal (x y : ProjectiveSpectrum 𝒜) :
    x.asHomogeneousIdeal ≤ y.asHomogeneousIdeal ↔ x ≤ y :=
  Iff.rfl


@[simp]
theorem as_ideal_lt_as_ideal (x y : ProjectiveSpectrum 𝒜) :
    x.asHomogeneousIdeal < y.asHomogeneousIdeal ↔ x < y :=
  Iff.rfl


theorem le_iff_mem_closure (x y : ProjectiveSpectrum 𝒜) :
    x ≤ y ↔ y ∈ closure ({x} : Set (ProjectiveSpectrum 𝒜)) := by
  rw [← as_ideal_le_as_ideal, ← zeroLocus_vanishingIdeal_eq_closure, mem_zeroLocus,
    vanishingIdeal_singleton]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    x y : ProjectiveSpectrum 𝒜
    ⊢ Iff (LE.le x.asHomogeneousIdeal y.asHomogeneousIdeal) (HasSubset.Subset ↑x.a …
  -/
  simp only [as_ideal_le_as_ideal, coe_subset_coe]
  /-
    🎉 no goals
  -/


