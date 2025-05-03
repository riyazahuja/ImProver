theorem algebraicIndependent_iff_ker_eq_bot :
    AlgebraicIndependent R x ↔
      RingHom.ker (MvPolynomial.aeval x : MvPolynomial ι R →ₐ[R] A).toRingHom = ⊥ :=
  RingHom.injective_iff_ker_eq_bot _


@[simp]
theorem algebraicIndependent_empty_type_iff [IsEmpty ι] :
    AlgebraicIndependent R x ↔ Injective (algebraMap R A) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : IsEmpty ι
    ⊢ Iff (AlgebraicIndependent R x) (Function.Injective ⇑(algebraMap R A))
  -/
  rw [algebraicIndependent_iff_injective_aeval, MvPolynomial.aeval_injective_iff_of_isEmpty]
  /-
    🎉 no goals
  -/


theorem algebraMap_injective : Injective (algebraMap R A) := by
  simpa [Function.comp_def] using
    (Injective.of_comp_iff (algebraicIndependent_iff_injective_aeval.1 hx) MvPolynomial.C).2
      (MvPolynomial.C_injective _ _)


theorem linearIndependent : LinearIndependent R x := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    ⊢ LinearIndependent R x
  -/
  rw [linearIndependent_iff_injective_linearCombination]
  have : Finsupp.linearCombination R x =
      (MvPolynomial.aeval x).toLinearMap.comp (Finsupp.linearCombination R X) := by
    ext
    simp
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    this : Eq (Finsupp.linearCombination R x) ((MvPolynomial.aeval x).toLinearMap. …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R x)
  -/
  rw [this]
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    this : Eq (Finsupp.linearCombination R x) ((MvPolynomial.aeval x).toLinearMap. …
    ⊢ Function.Injective ⇑((MvPolynomial.aeval x).toLinearMap.comp (Finsupp.linear …
  -/
  refine (algebraicIndependent_iff_injective_aeval.mp hx).comp ?_
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    this : Eq (Finsupp.linearCombination R x) ((MvPolynomial.aeval x).toLinearMap. …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R MvPolynomial.X)
  -/
  rw [← linearIndependent_iff_injective_linearCombination]
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    this : Eq (Finsupp.linearCombination R x) ((MvPolynomial.aeval x).toLinearMap. …
    ⊢ LinearIndependent R MvPolynomial.X
  -/
  exact linearIndependent_X _ _
  /-
    🎉 no goals
  -/


protected theorem injective [Nontrivial R] : Injective x :=
  hx.linearIndependent.injective


theorem ne_zero [Nontrivial R] (i : ι) : x i ≠ 0 :=
  hx.linearIndependent.ne_zero i


theorem map {f : A →ₐ[R] A'} (hf_inj : Set.InjOn f (adjoin R (range x))) :
    AlgebraicIndependent R (f ∘ x) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    hx : AlgebraicIndependent R x
    f : AlgHom R A A'
    hf_inj : Set.InjOn ⇑f ↑(Algebra.adjoin R (Set.range x))
    ⊢ AlgebraicIndependent R (Function.comp (⇑f) x)
  -/
  have : aeval (f ∘ x) = f.comp (aeval x) := by ext; simp
  have h : ∀ p : MvPolynomial ι R, aeval x p ∈ (@aeval R _ _ _ _ _ ((↑) : range x → A)).range := by
    intro p
    rw [AlgHom.mem_range]
    refine ⟨MvPolynomial.rename (codRestrict x (range x) mem_range_self) p, ?_⟩
    simp [Function.comp_def, aeval_rename]
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    hx : AlgebraicIndependent R x
    f : AlgHom R A A'
    hf_inj : Set.InjOn ⇑f ↑(Algebra.adjoin R (Set.range x))
    this : Eq (MvPolynomial.aeval (Function.comp (⇑f) x)) (f.comp (MvPolynomial.ae …
    h : ∀ (p : MvPolynomial ι R), Membership.mem (MvPolynomial.aeval Subtype.val). …
    ⊢ AlgebraicIndependent R (Function.comp (⇑f) x)
  -/
  intro x y hxy
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x✝ : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    hx : AlgebraicIndependent R x✝
    f : AlgHom R A A'
    hf_inj : Set.InjOn ⇑f ↑(Algebra.adjoin R (Set.range x✝))
    this : Eq (MvPolynomial.aeval (Function.comp (⇑f) x✝)) (f.comp (MvPolynomial.a …
    h : ∀ (p : MvPolynomial ι R), Membership.mem (MvPolynomial.aeval Subtype.val). …
    x y : MvPolynomial ι R
    hxy : Eq ((MvPolynomial.aeval (Function.comp (⇑f) x✝)) x) ((MvPolynomial.aeval …
    ⊢ Eq x y
  -/
  rw [this] at hxy
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x✝ : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    hx : AlgebraicIndependent R x✝
    f : AlgHom R A A'
    hf_inj : Set.InjOn ⇑f ↑(Algebra.adjoin R (Set.range x✝))
    this : Eq (MvPolynomial.aeval (Function.comp (⇑f) x✝)) (f.comp (MvPolynomial.a …
    h : ∀ (p : MvPolynomial ι R), Membership.mem (MvPolynomial.aeval Subtype.val). …
    x y : MvPolynomial ι R
    hxy : Eq ((f.comp (MvPolynomial.aeval x✝)) x) ((f.comp (MvPolynomial.aeval x✝) …
    ⊢ Eq x y
  -/
  rw [adjoin_eq_range] at hf_inj
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    A' : Type u_6
    x✝ : ι → A
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing A'
    inst✝¹ : Algebra R A
    inst✝ : Algebra R A'
    hx : AlgebraicIndependent R x✝
    f : AlgHom R A A'
    hf_inj : Set.InjOn ⇑f ↑(MvPolynomial.aeval Subtype.val).range
    this : Eq (MvPolynomial.aeval (Function.comp (⇑f) x✝)) (f.comp (MvPolynomial.a …
    h : ∀ (p : MvPolynomial ι R), Membership.mem (MvPolynomial.aeval Subtype.val). …
    x y : MvPolynomial ι R
    hxy : Eq ((f.comp (MvPolynomial.aeval x✝)) x) ((f.comp (MvPolynomial.aeval x✝) …
    ⊢ Eq x y
  -/
  exact hx (hf_inj (h x) (h y) hxy)
  /-
    🎉 no goals
  -/


theorem map' {f : A →ₐ[R] A'} (hf_inj : Injective f) : AlgebraicIndependent R (f ∘ x) :=
  hx.map hf_inj.injOn


/-- If `x = {x_i : A | i : ι}` and `f = {f_i : MvPolynomial ι R | i : ι}` are algebraically
independent over `R`, then `{f_i(x) | i : ι}` is also algebraically independent over `R`.
For the partial converse, see `AlgebraicIndependent.of_aeval`. -/
theorem aeval_of_algebraicIndependent
    {f : ι → MvPolynomial ι R} (hf : AlgebraicIndependent R f) :
    AlgebraicIndependent R fun i ↦ aeval x (f i) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    f : ι → MvPolynomial ι R
    hf : AlgebraicIndependent R f
    ⊢ AlgebraicIndependent R fun i => (MvPolynomial.aeval x) (f i)
  -/
  rw [algebraicIndependent_iff] at hx hf ⊢
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval x) p) 0 → Eq p 0
    f : ι → MvPolynomial ι R
    hf : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval f) p) 0 → Eq p 0
    ⊢ ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval fun i => (MvPolynomial.aev …
  -/
  intro p hp
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval x) p) 0 → Eq p 0
    f : ι → MvPolynomial ι R
    hf : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval f) p) 0 → Eq p 0
    p : MvPolynomial ι R
    hp : Eq ((MvPolynomial.aeval fun i => (MvPolynomial.aeval x) (f i)) p) 0
    ⊢ Eq p 0
  -/
  exact hf _ (hx _ (by rwa [← aeval_comp_bind₁, AlgHom.comp_apply] at hp))
  /-
    🎉 no goals
  -/


omit hx in
/-- If `{f_i(x) | i : ι}` is algebraically independent over `R`, then
`{f_i : MvPolynomial ι R | i : ι}` is also algebraically independent over `R`.
In fact, the `x = {x_i : A | i : ι}` is also transcendental over `R` provided that `R`
is a field and `ι` is finite; the proof needs transcendence degree. -/
theorem of_aeval {f : ι → MvPolynomial ι R}
    (H : AlgebraicIndependent R fun i ↦ aeval x (f i)) :
    AlgebraicIndependent R f := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : ι → MvPolynomial ι R
    H : AlgebraicIndependent R fun i => (MvPolynomial.aeval x) (f i)
    ⊢ AlgebraicIndependent R f
  -/
  rw [algebraicIndependent_iff] at H ⊢
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : ι → MvPolynomial ι R
    H : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval fun i => (MvPolynomial.a …
    ⊢ ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval f) p) 0 → Eq p 0
  -/
  intro p hp
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : ι → MvPolynomial ι R
    H : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval fun i => (MvPolynomial.a …
    p : MvPolynomial ι R
    hp : Eq ((MvPolynomial.aeval f) p) 0
    ⊢ Eq p 0
  -/
  exact H p (by rw [← aeval_comp_bind₁, AlgHom.comp_apply, bind₁, hp, map_zero])
  /-
    🎉 no goals
  -/


theorem MvPolynomial.algebraicIndependent_X (σ R : Type*) [CommRing R] :
    AlgebraicIndependent R (X (R := R) (σ := σ)) := by
  /-
    σ : Type u_7
    R : Type u_8
    inst✝ : CommRing R
    ⊢ AlgebraicIndependent R MvPolynomial.X
  -/
  rw [AlgebraicIndependent, aeval_X_left]
  /-
    σ : Type u_7
    R : Type u_8
    inst✝ : CommRing R
    ⊢ Function.Injective ⇑(AlgHom.id R (MvPolynomial σ R))
  -/
  exact injective_id
  /-
    🎉 no goals
  -/


theorem AlgHom.algebraicIndependent_iff (f : A →ₐ[R] A') (hf : Injective f) :
    AlgebraicIndependent R (f ∘ x) ↔ AlgebraicIndependent R x :=
  ⟨fun h => h.of_comp f, fun h => h.map hf.injOn⟩


@[nontriviality]
theorem algebraicIndependent_of_subsingleton [Subsingleton R] : AlgebraicIndependent R x :=
  algebraicIndependent_iff.2 fun _ _ => Subsingleton.elim _ _


theorem algebraicIndependent_adjoin (hs : AlgebraicIndependent R x) :
    @AlgebraicIndependent ι R (adjoin R (range x))
      (fun i : ι => ⟨x i, subset_adjoin (mem_range_self i)⟩) _ _ _ :=
  AlgebraicIndependent.of_comp (adjoin R (range x)).val hs


/-- A set of algebraically independent elements in an algebra `A` over a ring `K` is also
algebraically independent over a subring `R` of `K`. -/
theorem AlgebraicIndependent.restrictScalars {K : Type*} [CommRing K] [Algebra R K] [Algebra K A]
    [IsScalarTower R K A] (hinj : Function.Injective (algebraMap R K))
    (ai : AlgebraicIndependent K x) : AlgebraicIndependent R x := by
  have : (aeval x : MvPolynomial ι K →ₐ[K] A).toRingHom.comp (MvPolynomial.map (algebraMap R K)) =
      (aeval x : MvPolynomial ι R →ₐ[R] A).toRingHom := by
    ext <;> simp [algebraMap_eq_smul_one]
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    K : Type u_7
    inst✝³ : CommRing K
    inst✝² : Algebra R K
    inst✝¹ : Algebra K A
    inst✝ : IsScalarTower R K A
    hinj : Function.Injective ⇑(algebraMap R K)
    ai : AlgebraicIndependent K x
    this : Eq ((MvPolynomial.aeval x).comp (MvPolynomial.map (algebraMap R K))) (M …
    ⊢ AlgebraicIndependent R x
  -/
  show Injective (aeval x).toRingHom
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    K : Type u_7
    inst✝³ : CommRing K
    inst✝² : Algebra R K
    inst✝¹ : Algebra K A
    inst✝ : IsScalarTower R K A
    hinj : Function.Injective ⇑(algebraMap R K)
    ai : AlgebraicIndependent K x
    this : Eq ((MvPolynomial.aeval x).comp (MvPolynomial.map (algebraMap R K))) (M …
    ⊢ Function.Injective ⇑(MvPolynomial.aeval x).toRingHom
  -/
  rw [← this, RingHom.coe_comp]
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : Algebra R A
    K : Type u_7
    inst✝³ : CommRing K
    inst✝² : Algebra R K
    inst✝¹ : Algebra K A
    inst✝ : IsScalarTower R K A
    hinj : Function.Injective ⇑(algebraMap R K)
    ai : AlgebraicIndependent K x
    this : Eq ((MvPolynomial.aeval x).comp (MvPolynomial.map (algebraMap R K))) (M …
    ⊢ Function.Injective (Function.comp ⇑(MvPolynomial.aeval x).toRingHom ⇑(MvPoly …
  -/
  exact Injective.comp ai (MvPolynomial.map_injective _ hinj)
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.of_ringHom_of_comp_eq (H : AlgebraicIndependent S (g ∘ x))
    (hf : Function.Injective f)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    AlgebraicIndependent R x := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    S : Type u_7
    B : Type u_8
    FRS : Type u_9
    FAB : Type u_10
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra S B
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : AlgebraicIndependent S (Function.comp (⇑g) x)
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ AlgebraicIndependent R x
  -/
  rw [algebraicIndependent_iff] at H ⊢
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    S : Type u_7
    B : Type u_8
    FRS : Type u_9
    FAB : Type u_10
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra S B
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : ∀ (p : MvPolynomial ι S), Eq ((MvPolynomial.aeval (Function.comp (⇑g) x))  …
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval x) p) 0 → Eq p 0
  -/
  intro p hp
  have := H (p.map f) <| by
    have : (g : A →+* B) _ = _ := congr(g $hp)
    rwa [map_zero, map_aeval, ← h, ← eval₂Hom_map_hom, ← aeval_eq_eval₂Hom] at this
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    S : Type u_7
    B : Type u_8
    FRS : Type u_9
    FAB : Type u_10
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra S B
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : ∀ (p : MvPolynomial ι S), Eq ((MvPolynomial.aeval (Function.comp (⇑g) x))  …
    hf : Function.Injective ⇑f
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    p : MvPolynomial ι R
    hp : Eq ((MvPolynomial.aeval x) p) 0
    this : Eq ((MvPolynomial.map ↑f) p) 0
    ⊢ Eq p 0
  -/
  exact map_injective (f : R →+* S) hf (by rwa [map_zero])
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.ringHom_of_comp_eq (H : AlgebraicIndependent R x)
    (hf : Function.Surjective f) (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    AlgebraicIndependent S (g ∘ x) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    S : Type u_7
    B : Type u_8
    FRS : Type u_9
    FAB : Type u_10
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra S B
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : AlgebraicIndependent R x
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ AlgebraicIndependent S (Function.comp (⇑g) x)
  -/
  rw [algebraicIndependent_iff] at H ⊢
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    S : Type u_7
    B : Type u_8
    FRS : Type u_9
    FAB : Type u_10
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra S B
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval x) p) 0 → Eq p 0
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    ⊢ ∀ (p : MvPolynomial ι S), Eq ((MvPolynomial.aeval (Function.comp (⇑g) x)) p) …
  -/
  intro p hp
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing A
    inst✝⁷ : Algebra R A
    S : Type u_7
    B : Type u_8
    FRS : Type u_9
    FAB : Type u_10
    inst✝⁶ : CommRing S
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra S B
    inst✝³ : FunLike FRS R S
    inst✝² : RingHomClass FRS R S
    inst✝¹ : FunLike FAB A B
    inst✝ : RingHomClass FAB A B
    f : FRS
    g : FAB
    H : ∀ (p : MvPolynomial ι R), Eq ((MvPolynomial.aeval x) p) 0 → Eq p 0
    hf : Function.Surjective ⇑f
    hg : Function.Injective ⇑g
    h : Eq ((algebraMap S B).comp ↑f) ((↑g).comp (algebraMap R A))
    p : MvPolynomial ι S
    hp : Eq ((MvPolynomial.aeval (Function.comp (⇑g) x)) p) 0
    ⊢ Eq p 0
  -/
  obtain ⟨q, rfl⟩ := map_surjective (f : R →+* S) hf p
  rw [H q (hg (by rwa [map_zero, ← RingHom.coe_coe g, map_aeval, ← h, ← eval₂Hom_map_hom,
    ← aeval_eq_eval₂Hom])), map_zero]


theorem algebraicIndependent_ringHom_iff_of_comp_eq
    (hg : Function.Injective g)
    (h : RingHom.comp (algebraMap S B) f = RingHom.comp g (algebraMap R A)) :
    AlgebraicIndependent S (g ∘ x) ↔ AlgebraicIndependent R x :=
  ⟨fun H ↦ H.of_ringHom_of_comp_eq f g (EquivLike.injective f) h,
    fun H ↦ H.ringHom_of_comp_eq f g (EquivLike.surjective f) hg h⟩


/-- Every finite subset of an algebraically independent set is algebraically independent. -/
theorem algebraicIndependent_finset_map_embedding_subtype (s : Set A)
    (li : AlgebraicIndependent R ((↑) : s → A)) (t : Finset s) :
    AlgebraicIndependent R ((↑) : Finset.map (Embedding.subtype s) t → A) := by
  let f : t.map (Embedding.subtype s) → s := fun x =>
    ⟨x.1, by
      obtain ⟨x, h⟩ := x
      rw [Finset.mem_map] at h
      obtain ⟨a, _, rfl⟩ := h
      simp only [Subtype.coe_prop, Embedding.coe_subtype]⟩
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  convert AlgebraicIndependent.comp li f _
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    ⊢ Function.Injective f
  -/
  rintro ⟨x, hx⟩ ⟨y, hy⟩
  /-
    case mk.mk
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    x : A
    hx : Membership.mem (Finset.map (Function.Embedding.subtype s) t) x
    y : A
    hy : Membership.mem (Finset.map (Function.Embedding.subtype s) t) y
    ⊢ Eq (f ⟨x, hx⟩) (f ⟨y, hy⟩) → Eq ⟨x, hx⟩ ⟨y, hy⟩
  -/
  rw [Finset.mem_map] at hx hy
  /-
    case mk.mk
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    x : A
    hx✝ : Membership.mem (Finset.map (Function.Embedding.subtype s) t) x
    hx : Exists fun a => And (Membership.mem t a) (Eq ((Function.Embedding.subtype …
    y : A
    hy✝ : Membership.mem (Finset.map (Function.Embedding.subtype s) t) y
    hy : Exists fun a => And (Membership.mem t a) (Eq ((Function.Embedding.subtype …
    ⊢ Eq (f ⟨x, hx✝⟩) (f ⟨y, hy✝⟩) → Eq ⟨x, hx✝⟩ ⟨y, hy✝⟩
  -/
  obtain ⟨a, _, rfl⟩ := hx
  /-
    case mk.mk.intro.intro
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    y : A
    hy✝ : Membership.mem (Finset.map (Function.Embedding.subtype s) t) y
    hy : Exists fun a => And (Membership.mem t a) (Eq ((Function.Embedding.subtype …
    a : Subtype s
    left✝ : Membership.mem t a
    hx : Membership.mem (Finset.map (Function.Embedding.subtype s) t) ((Function.E …
    ⊢ Eq (f ⟨(Function.Embedding.subtype s) a, hx⟩) (f ⟨y, hy✝⟩) → Eq ⟨(Function.E …
  -/
  obtain ⟨b, _, rfl⟩ := hy
  /-
    case mk.mk.intro.intro.intro.intro
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    f : (Subtype fun x => Membership.mem (Finset.map (Function.Embedding.subtype s …
    a : Subtype s
    left✝¹ : Membership.mem t a
    hx : Membership.mem (Finset.map (Function.Embedding.subtype s) t) ((Function.E …
    b : Subtype s
    left✝ : Membership.mem t b
    hy : Membership.mem (Finset.map (Function.Embedding.subtype s) t) ((Function.E …
    ⊢ Eq (f ⟨(Function.Embedding.subtype s) a, hx⟩) (f ⟨(Function.Embedding.subtyp …
  -/
  simp only [f, imp_self, Subtype.mk_eq_mk]
  /-
    🎉 no goals
  -/


/-- If every finite set of algebraically independent element has cardinality at most `n`,
then the same is true for arbitrary sets of algebraically independent elements. -/
theorem algebraicIndependent_bounded_of_finset_algebraicIndependent_bounded {n : ℕ}
    (H : ∀ s : Finset A, (AlgebraicIndependent R fun i : s => (i : A)) → s.card ≤ n) :
    ∀ s : Set A, AlgebraicIndependent R ((↑) : s → A) → Cardinal.mk s ≤ n := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    n : Nat
    H : ∀ (s : Finset A), (AlgebraicIndependent R fun i => ↑i) → LE.le s.card n
    ⊢ ∀ (s : Set A), AlgebraicIndependent R Subtype.val → LE.le (Cardinal.mk ↑s) ↑n
  -/
  intro s li
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    n : Nat
    H : ∀ (s : Finset A), (AlgebraicIndependent R fun i => ↑i) → LE.le s.card n
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    ⊢ LE.le (Cardinal.mk ↑s) ↑n
  -/
  apply Cardinal.card_le_of
  /-
    case H
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    n : Nat
    H : ∀ (s : Finset A), (AlgebraicIndependent R fun i => ↑i) → LE.le s.card n
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    ⊢ ∀ (s_1 : Finset ↑s), LE.le s_1.card n
  -/
  intro t
  /-
    case H
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    n : Nat
    H : ∀ (s : Finset A), (AlgebraicIndependent R fun i => ↑i) → LE.le s.card n
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    ⊢ LE.le t.card n
  -/
  rw [← Finset.card_map (Embedding.subtype s)]
  /-
    case H
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    n : Nat
    H : ∀ (s : Finset A), (AlgebraicIndependent R fun i => ↑i) → LE.le s.card n
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    ⊢ LE.le (Finset.map (Function.Embedding.subtype s) t).card n
  -/
  apply H
  /-
    case H.a
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    n : Nat
    H : ∀ (s : Finset A), (AlgebraicIndependent R fun i => ↑i) → LE.le s.card n
    s : Set A
    li : AlgebraicIndependent R Subtype.val
    t : Finset ↑s
    ⊢ AlgebraicIndependent R fun i => ↑i
  -/
  apply algebraicIndependent_finset_map_embedding_subtype _ li
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.restrict_of_comp_subtype {s : Set ι}
    (hs : AlgebraicIndependent R (x ∘ (↑) : s → A)) : AlgebraicIndependent R (s.restrict x) :=
  hs


theorem algebraicIndependent_empty_iff :
                                                                                      /-
                                                                                        R : Type u_3
                                                                                        A : Type u_5
                                                                                        inst✝² : CommRing R
                                                                                        inst✝¹ : CommRing A
                                                                                        inst✝ : Algebra R A
                                                                                        ⊢ Iff (AlgebraicIndependent R Subtype.val) (Function.Injective ⇑(algebraMap R  …
                                                                                      -/
    AlgebraicIndependent R ((↑) : (∅ : Set A) → A) ↔ Injective (algebraMap R A) := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem AlgebraicIndependent.to_subtype_range {ι} {f : ι → A} (hf : AlgebraicIndependent R f) :
    AlgebraicIndependent R ((↑) : range f → A) := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Type u_7
    f : ι → A
    hf : AlgebraicIndependent R f
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  nontriviality R
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Type u_7
    f : ι → A
    hf : AlgebraicIndependent R f
    a✝ : Nontrivial R
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  rwa [algebraicIndependent_subtype_range hf.injective]
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.to_subtype_range' {ι} {f : ι → A} (hf : AlgebraicIndependent R f) {t}
    (ht : range f = t) : AlgebraicIndependent R ((↑) : t → A) :=
  ht ▸ hf.to_subtype_range


theorem algebraicIndependent_comp_subtype {s : Set ι} :
    AlgebraicIndependent R (x ∘ (↑) : s → A) ↔
      ∀ p ∈ MvPolynomial.supported R s, aeval x p = 0 → p = 0 := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set ι
    ⊢ Iff (AlgebraicIndependent R (Function.comp x Subtype.val)) (∀ (p : MvPolynom …
  -/
  have : (aeval (x ∘ (↑) : s → A) : _ →ₐ[R] _) = (aeval x).comp (rename (↑)) := by ext; simp
  have : ∀ p : MvPolynomial s R, rename ((↑) : s → ι) p = 0 ↔ p = 0 :=
    (injective_iff_map_eq_zero' (rename ((↑) : s → ι) : MvPolynomial s R →ₐ[R] _).toRingHom).1
      (rename_injective _ Subtype.val_injective)
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set ι
    this✝ : Eq (MvPolynomial.aeval (Function.comp x Subtype.val)) ((MvPolynomial.a …
    this : ∀ (p : MvPolynomial (↑s) R), Iff (Eq ((MvPolynomial.rename Subtype.val) …
    ⊢ Iff (AlgebraicIndependent R (Function.comp x Subtype.val)) (∀ (p : MvPolynom …
  -/
  simp [algebraicIndependent_iff, supported_eq_range_rename, *]
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_subtype {s : Set A} :
    AlgebraicIndependent R ((↑) : s → A) ↔
      ∀ p : MvPolynomial A R, p ∈ MvPolynomial.supported R s → aeval id p = 0 → p = 0 := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set A
    ⊢ Iff (AlgebraicIndependent R Subtype.val) (∀ (p : MvPolynomial A R), Membersh …
  -/
  apply @algebraicIndependent_comp_subtype _ _ _ id
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_of_finite (s : Set A)
    (H : ∀ t ⊆ s, t.Finite → AlgebraicIndependent R ((↑) : t → A)) :
    AlgebraicIndependent R ((↑) : s → A) :=
  algebraicIndependent_subtype.2 fun p hp ↦
                                                                                            /-
                                                                                              R : Type u_3
                                                                                              A : Type u_5
                                                                                              inst✝² : CommRing R
                                                                                              inst✝¹ : CommRing A
                                                                                              inst✝ : Algebra R A
                                                                                              s : Set A
                                                                                              H : ∀ (t : Set A), HasSubset.Subset t s → t.Finite → AlgebraicIndependent R Su …
                                                                                              p : MvPolynomial A R
                                                                                              hp : Membership.mem (MvPolynomial.supported R s) p
                                                                                              ⊢ Membership.mem (MvPolynomial.supported R ↑p.vars) p
                                                                                            -/
    algebraicIndependent_subtype.1 (H _ (mem_supported.1 hp) (Finset.finite_toSet _)) _ (by simp)
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem algebraicIndependent_of_finite_type
    (H : ∀ t : Set ι, t.Finite → AlgebraicIndependent R fun i : t ↦ x i) :
    AlgebraicIndependent R x :=
  (injective_iff_map_eq_zero _).mpr fun p ↦
    algebraicIndependent_comp_subtype.1 (H _ p.vars.finite_toSet) _ p.mem_supported_vars


theorem AlgebraicIndependent.image_of_comp {ι ι'} (s : Set ι) (f : ι → ι') (g : ι' → A)
    (hs : AlgebraicIndependent R fun x : s => g (f x)) :
    AlgebraicIndependent R fun x : f '' s => g x := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Type u_7
    ι' : Type u_8
    s : Set ι
    f : ι → ι'
    g : ι' → A
    hs : AlgebraicIndependent R fun x => g (f ↑x)
    ⊢ AlgebraicIndependent R fun x => g ↑x
  -/
  nontriviality R
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Type u_7
    ι' : Type u_8
    s : Set ι
    f : ι → ι'
    g : ι' → A
    hs : AlgebraicIndependent R fun x => g (f ↑x)
    a✝ : Nontrivial R
    ⊢ AlgebraicIndependent R fun x => g ↑x
  -/
  have : InjOn f s := injOn_iff_injective.2 hs.injective.of_comp
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Type u_7
    ι' : Type u_8
    s : Set ι
    f : ι → ι'
    g : ι' → A
    hs : AlgebraicIndependent R fun x => g (f ↑x)
    a✝ : Nontrivial R
    this : Set.InjOn f s
    ⊢ AlgebraicIndependent R fun x => g ↑x
  -/
  exact (algebraicIndependent_equiv' (Equiv.Set.imageOfInjOn f s this) rfl).1 hs
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.image {ι} {s : Set ι} {f : ι → A}
    (hs : AlgebraicIndependent R fun x : s => f x) :
    AlgebraicIndependent R fun x : f '' s => (x : A) := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Type u_7
    s : Set ι
    f : ι → A
    hs : AlgebraicIndependent R fun x => f ↑x
    ⊢ AlgebraicIndependent R fun x => ↑x
  -/
  convert AlgebraicIndependent.image_of_comp s f id hs
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_iUnion_of_directed {η : Type*} [Nonempty η] {s : η → Set A}
    (hs : Directed (· ⊆ ·) s) (h : ∀ i, AlgebraicIndependent R ((↑) : s i → A)) :
    AlgebraicIndependent R ((↑) : (⋃ i, s i) → A) := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    η : Type u_7
    inst✝ : Nonempty η
    s : η → Set A
    hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (i : η), AlgebraicIndependent R Subtype.val
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  refine algebraicIndependent_of_finite (⋃ i, s i) fun t ht ft => ?_
  /-
    R : Type u_3
    A : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    η : Type u_7
    inst✝ : Nonempty η
    s : η → Set A
    hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (i : η), AlgebraicIndependent R Subtype.val
    t : Set A
    ht : HasSubset.Subset t (Set.iUnion fun i => s i)
    ft : t.Finite
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  rcases finite_subset_iUnion ft ht with ⟨I, fi, hI⟩
  /-
    case intro.intro
    R : Type u_3
    A : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    η : Type u_7
    inst✝ : Nonempty η
    s : η → Set A
    hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (i : η), AlgebraicIndependent R Subtype.val
    t : Set A
    ht : HasSubset.Subset t (Set.iUnion fun i => s i)
    ft : t.Finite
    I : Set η
    fi : I.Finite
    hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  rcases hs.finset_le fi.toFinset with ⟨i, hi⟩
  /-
    case intro.intro.intro
    R : Type u_3
    A : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    η : Type u_7
    inst✝ : Nonempty η
    s : η → Set A
    hs : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (i : η), AlgebraicIndependent R Subtype.val
    t : Set A
    ht : HasSubset.Subset t (Set.iUnion fun i => s i)
    ft : t.Finite
    I : Set η
    fi : I.Finite
    hI : HasSubset.Subset t (Set.iUnion fun i => Set.iUnion fun h => s i)
    i : η
    hi : ∀ (i_1 : η), Membership.mem fi.toFinset i_1 → HasSubset.Subset (s i_1) (s …
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  exact (h i).mono (Subset.trans hI <| iUnion₂_subset fun j hj => hi j (fi.mem_toFinset.2 hj))
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_sUnion_of_directed {s : Set (Set A)} (hsn : s.Nonempty)
    (hs : DirectedOn (· ⊆ ·) s) (h : ∀ a ∈ s, AlgebraicIndependent R ((↑) : a → A)) :
    AlgebraicIndependent R ((↑) : ⋃₀ s → A) := by
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set (Set A)
    hsn : s.Nonempty
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set A), Membership.mem s a → AlgebraicIndependent R Subtype.val
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  letI : Nonempty s := Nonempty.to_subtype hsn
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set (Set A)
    hsn : s.Nonempty
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set A), Membership.mem s a → AlgebraicIndependent R Subtype.val
    this : Nonempty ↑s := Set.Nonempty.to_subtype hsn
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  rw [sUnion_eq_iUnion]
  /-
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s : Set (Set A)
    hsn : s.Nonempty
    hs : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    h : ∀ (a : Set A), Membership.mem s a → AlgebraicIndependent R Subtype.val
    this : Nonempty ↑s := Set.Nonempty.to_subtype hsn
    ⊢ AlgebraicIndependent R Subtype.val
  -/
  exact algebraicIndependent_iUnion_of_directed hs.directed_val (by simpa using h)
  /-
    🎉 no goals
  -/


theorem exists_maximal_algebraicIndependent (s t : Set A) (hst : s ⊆ t)
    (hs : AlgebraicIndependent R ((↑) : s → A)) : ∃ u, s ⊆ u ∧
      Maximal (fun (x : Set A) ↦ AlgebraicIndependent R ((↑) : x → A) ∧ x ⊆ t) u := by
  refine zorn_subset_nonempty { u : Set A | AlgebraicIndependent R ((↑) : u → A) ∧ u ⊆ t}
    (fun c hc chainc hcn ↦ ⟨⋃₀ c, ⟨?_, ?_⟩, fun _ ↦ subset_sUnion_of_mem⟩) s ⟨hs, hst⟩
    /-
      case refine_1
      R : Type u_3
      A : Type u_5
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      s t : Set A
      hst : HasSubset.Subset s t
      hs : AlgebraicIndependent R Subtype.val
      c : Set (Set A)
      hc : HasSubset.Subset c (setOf fun u => And (AlgebraicIndependent R Subtype.va …
      chainc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
      hcn : c.Nonempty
      ⊢ AlgebraicIndependent R Subtype.val
    -/
  · exact algebraicIndependent_sUnion_of_directed hcn chainc.directedOn (fun x hxc ↦ (hc hxc).1)
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    s t : Set A
    hst : HasSubset.Subset s t
    hs : AlgebraicIndependent R Subtype.val
    c : Set (Set A)
    hc : HasSubset.Subset c (setOf fun u => And (AlgebraicIndependent R Subtype.va …
    chainc : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) c
    hcn : c.Nonempty
    ⊢ HasSubset.Subset c.sUnion t
  -/
  exact fun x ⟨w, hyc, hwy⟩ ↦ (hc hyc).2 hwy
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.repr_ker (hx : AlgebraicIndependent R x) :
    RingHom.ker (hx.repr : adjoin R (range x) →+* MvPolynomial ι R) = ⊥ :=
  (RingHom.injective_iff_ker_eq_bot _).1 (AlgEquiv.injective _)

-- TODO - make this an `AlgEquiv`

/-- The isomorphism between `MvPolynomial (Option ι) R` and the polynomial ring over
the algebra generated by an algebraically independent family. -/
def AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin (hx : AlgebraicIndependent R x) :
    MvPolynomial (Option ι) R ≃+* Polynomial (adjoin R (Set.range x)) :=
  (MvPolynomial.optionEquivLeft _ _).toRingEquiv.trans
    (Polynomial.mapEquiv hx.aevalEquiv.toRingEquiv)


@[simp]
theorem AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_apply
    (hx : AlgebraicIndependent R x) (y) :
    hx.mvPolynomialOptionEquivPolynomialAdjoin y =
      Polynomial.map (hx.aevalEquiv : MvPolynomial ι R →+* adjoin R (range x))
        (aeval (fun o : Option ι => o.elim Polynomial.X fun s : ι => Polynomial.C (X s)) y) :=
  rfl


/-- `simp`-normal form of `mvPolynomialOptionEquivPolynomialAdjoin_C` -/
@[simp]
theorem AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_C'
    (hx : AlgebraicIndependent R x) (r) :
    Polynomial.C (hx.aevalEquiv (C r)) = Polynomial.C (algebraMap _ _ r) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    r : R
    ⊢ Eq (Polynomial.C (hx.aevalEquiv (MvPolynomial.C r))) (Polynomial.C ((algebra …
  -/
  congr
  /-
    case h.e_6.h
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    r : R
    ⊢ Eq (hx.aevalEquiv (MvPolynomial.C r)) ((algebraMap R (Subtype fun x_1 => Mem …
  -/
  apply_fun Subtype.val using Subtype.val_injective
  /-
    case h.e_6.h
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    r : R
    ⊢ Eq ↑(hx.aevalEquiv (MvPolynomial.C r)) ↑((algebraMap R (Subtype fun x_1 => M …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_C
    (hx : AlgebraicIndependent R x) (r) :
    hx.mvPolynomialOptionEquivPolynomialAdjoin (C r) = Polynomial.C (algebraMap _ _ r) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    r : R
    ⊢ Eq (hx.mvPolynomialOptionEquivPolynomialAdjoin (MvPolynomial.C r)) (Polynomi …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_X_none
    (hx : AlgebraicIndependent R x) :
    hx.mvPolynomialOptionEquivPolynomialAdjoin (X none) = Polynomial.X := by
  rw [AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_apply, aeval_X, Option.elim,
    Polynomial.map_X]


theorem AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_X_some
    (hx : AlgebraicIndependent R x) (i) :
    hx.mvPolynomialOptionEquivPolynomialAdjoin (X (some i)) =
      Polynomial.C (hx.aevalEquiv (X i)) := by
  rw [AlgebraicIndependent.mvPolynomialOptionEquivPolynomialAdjoin_apply, aeval_X, Option.elim,
    Polynomial.map_C, RingHom.coe_coe]


theorem AlgebraicIndependent.aeval_comp_mvPolynomialOptionEquivPolynomialAdjoin
    (hx : AlgebraicIndependent R x) (a : A) :
    RingHom.comp
        (↑(Polynomial.aeval a : Polynomial (adjoin R (Set.range x)) →ₐ[_] A) :
          Polynomial (adjoin R (Set.range x)) →+* A)
        hx.mvPolynomialOptionEquivPolynomialAdjoin.toRingHom =
      ↑(MvPolynomial.aeval fun o : Option ι => o.elim a x : MvPolynomial (Option ι) R →ₐ[R] A) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    hx : AlgebraicIndependent R x
    a : A
    ⊢ Eq ((↑(Polynomial.aeval a)).comp hx.mvPolynomialOptionEquivPolynomialAdjoin. …
  -/
  refine MvPolynomial.ringHom_ext ?_ ?_ <;>
    simp only [RingHom.comp_apply, RingEquiv.toRingHom_eq_coe, RingEquiv.coe_toRingHom,
      AlgHom.coe_toRingHom, AlgHom.coe_toRingHom]
    /-
      case refine_1
      ι : Type u_1
      R : Type u_3
      A : Type u_5
      x : ι → A
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hx : AlgebraicIndependent R x
      a : A
      ⊢ ∀ (r : R), Eq ((Polynomial.aeval a) (hx.mvPolynomialOptionEquivPolynomialAdj …
    -/
  · intro r
    rw [hx.mvPolynomialOptionEquivPolynomialAdjoin_C, aeval_C, Polynomial.aeval_C,
      IsScalarTower.algebraMap_apply R (adjoin R (range x)) A]
    /-
      case refine_2
      ι : Type u_1
      R : Type u_3
      A : Type u_5
      x : ι → A
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      hx : AlgebraicIndependent R x
      a : A
      ⊢ ∀ (i : Option ι), Eq ((Polynomial.aeval a) (hx.mvPolynomialOptionEquivPolyno …
    -/
  · rintro (⟨⟩ | ⟨i⟩)
    · rw [hx.mvPolynomialOptionEquivPolynomialAdjoin_X_none, aeval_X, Polynomial.aeval_X,
        Option.elim]
    · rw [hx.mvPolynomialOptionEquivPolynomialAdjoin_X_some, Polynomial.aeval_C,
        hx.algebraMap_aevalEquiv, aeval_X, aeval_X, Option.elim]


theorem algebraicIndependent_empty_type [IsEmpty ι] [Nontrivial A] : AlgebraicIndependent K x := by
  /-
    ι : Type u_1
    K : Type u_4
    A : Type u_5
    x : ι → A
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra K A
    inst✝¹ : IsEmpty ι
    inst✝ : Nontrivial A
    ⊢ AlgebraicIndependent K x
  -/
  rw [algebraicIndependent_empty_type_iff]
  /-
    ι : Type u_1
    K : Type u_4
    A : Type u_5
    x : ι → A
    inst✝⁴ : CommRing A
    inst✝³ : Field K
    inst✝² : Algebra K A
    inst✝¹ : IsEmpty ι
    inst✝ : Nontrivial A
    ⊢ Function.Injective ⇑(algebraMap K A)
  -/
  exact RingHom.injective _
  /-
    🎉 no goals
  -/


theorem algebraicIndependent_empty [Nontrivial A] :
    AlgebraicIndependent K ((↑) : (∅ : Set A) → A) :=
  algebraicIndependent_empty_type


