/-- `I.map f` is the span of the image of the ideal `I` under `f`, which may be bigger than
  the image itself. -/
def map (I : Ideal R) : Ideal S :=
  span (f '' I)


/-- `I.comap f` is the preimage of `I` under `f`. -/
def comap [RingHomClass F R S] (I : Ideal S) : Ideal R where
  carrier := f ⁻¹' I
  add_mem' {x y} hx hy := by
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      f : F
      I✝ J : Ideal R
      K L : Ideal S
      inst✝ : RingHomClass F R S
      I : Ideal S
      x y : R
      hx : Membership.mem (Set.preimage ⇑f ↑I) x
      hy : Membership.mem (Set.preimage ⇑f ↑I) y
      ⊢ Membership.mem (Set.preimage ⇑f ↑I) (HAdd.hAdd x y)
    -/
    simp only [Set.mem_preimage, SetLike.mem_coe, map_add f] at hx hy ⊢
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      f : F
      I✝ J : Ideal R
      K L : Ideal S
      inst✝ : RingHomClass F R S
      I : Ideal S
      x y : R
      hx : Membership.mem I (f x)
      hy : Membership.mem I (f y)
      ⊢ Membership.mem I (HAdd.hAdd (f x) (f y))
    -/
    exact add_mem hx hy
    /-
      🎉 no goals
    -/
                  /-
                    R : Type u
                    S : Type v
                    F : Type u_1
                    inst✝³ : Semiring R
                    inst✝² : Semiring S
                    inst✝¹ : FunLike F R S
                    f : F
                    I✝ J : Ideal R
                    K L : Ideal S
                    inst✝ : RingHomClass F R S
                    I : Ideal S
                    ⊢ Membership.mem { carrier := Set.preimage ⇑f ↑I, add_mem' := ⋯ }.carrier 0
                  -/
  zero_mem' := by simp only [Set.mem_preimage, map_zero, SetLike.mem_coe, Submodule.zero_mem]
                  /-
                    🎉 no goals
                  -/
  smul_mem' c x hx := by
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      f : F
      I✝ J : Ideal R
      K L : Ideal S
      inst✝ : RingHomClass F R S
      I : Ideal S
      c x : R
      hx : Membership.mem { carrier := Set.preimage ⇑f ↑I, add_mem' := ⋯, zero_mem'  …
      ⊢ Membership.mem { carrier := Set.preimage ⇑f ↑I, add_mem' := ⋯, zero_mem' :=  …
    -/
    simp only [smul_eq_mul, Set.mem_preimage, map_mul, SetLike.mem_coe] at *
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      f : F
      I✝ J : Ideal R
      K L : Ideal S
      inst✝ : RingHomClass F R S
      I : Ideal S
      c x : R
      hx : Membership.mem I (f x)
      ⊢ Membership.mem I (HMul.hMul (f c) (f x))
    -/
    exact mul_mem_left I _ hx
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_comap [RingHomClass F R S] (I : Ideal S) : (comap f I : Set R) = f ⁻¹' I := rfl


lemma comap_coe [RingHomClass F R S] (I : Ideal S) : I.comap (f : R →+* S) = I.comap f := rfl


theorem map_mono (h : I ≤ J) : map f I ≤ map f J :=
  span_mono <| Set.image_subset _ h


theorem mem_map_of_mem (f : F) {I : Ideal R} {x : R} (h : x ∈ I) : f x ∈ map f I :=
  subset_span ⟨x, h, rfl⟩


theorem apply_coe_mem_map (f : F) (I : Ideal R) (x : I) : f x ∈ I.map f :=
  mem_map_of_mem f x.2


theorem map_le_iff_le_comap [RingHomClass F R S] : map f I ≤ K ↔ I ≤ comap f K :=
  span_le.trans Set.image_subset_iff


@[simp]
theorem mem_comap [RingHomClass F R S] {x} : x ∈ comap f K ↔ f x ∈ K :=
  Iff.rfl


theorem comap_mono [RingHomClass F R S] (h : K ≤ L) : comap f K ≤ comap f L :=
  Set.preimage_mono fun _ hx => h hx


theorem comap_ne_top [RingHomClass F R S] (hK : K ≠ ⊤) : comap f K ≠ ⊤ :=
                             /-
                               R : Type u
                               S : Type v
                               F : Type u_1
                               inst✝³ : Semiring R
                               inst✝² : Semiring S
                               inst✝¹ : FunLike F R S
                               f : F
                               K : Ideal S
                               inst✝ : RingHomClass F R S
                               hK : Ne K Top.top
                               ⊢ Not (Membership.mem (Ideal.comap f K) 1)
                             -/
  (ne_top_iff_one _).2 <| by rw [mem_comap, map_one]; exact (ne_top_iff_one _).1 hK
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem map_le_comap_of_inv_on [RingHomClass G S R] (g : G) (I : Ideal R)
    (hf : Set.LeftInvOn g f I) :
    I.map f ≤ I.comap g := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝⁴ : Semiring R
    inst✝³ : Semiring S
    inst✝² : FunLike F R S
    f : F
    G : Type u_2
    inst✝¹ : FunLike G S R
    inst✝ : RingHomClass G S R
    g : G
    I : Ideal R
    hf : Set.LeftInvOn ⇑g ⇑f ↑I
    ⊢ LE.le (Ideal.map f I) (Ideal.comap g I)
  -/
  refine Ideal.span_le.2 ?_
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝⁴ : Semiring R
    inst✝³ : Semiring S
    inst✝² : FunLike F R S
    f : F
    G : Type u_2
    inst✝¹ : FunLike G S R
    inst✝ : RingHomClass G S R
    g : G
    I : Ideal R
    hf : Set.LeftInvOn ⇑g ⇑f ↑I
    ⊢ HasSubset.Subset (Set.image ⇑f ↑I) ↑(Ideal.comap g I)
  -/
  rintro x ⟨x, hx, rfl⟩
  /-
    case intro.intro
    R : Type u
    S : Type v
    F : Type u_1
    inst✝⁴ : Semiring R
    inst✝³ : Semiring S
    inst✝² : FunLike F R S
    f : F
    G : Type u_2
    inst✝¹ : FunLike G S R
    inst✝ : RingHomClass G S R
    g : G
    I : Ideal R
    hf : Set.LeftInvOn ⇑g ⇑f ↑I
    x : R
    hx : Membership.mem (↑I) x
    ⊢ Membership.mem (↑(Ideal.comap g I)) (f x)
  -/
  rw [SetLike.mem_coe, mem_comap, hf hx]
  /-
    case intro.intro
    R : Type u
    S : Type v
    F : Type u_1
    inst✝⁴ : Semiring R
    inst✝³ : Semiring S
    inst✝² : FunLike F R S
    f : F
    G : Type u_2
    inst✝¹ : FunLike G S R
    inst✝ : RingHomClass G S R
    g : G
    I : Ideal R
    hf : Set.LeftInvOn ⇑g ⇑f ↑I
    x : R
    hx : Membership.mem (↑I) x
    ⊢ Membership.mem I x
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem comap_le_map_of_inv_on [RingHomClass F R S] (g : G) (I : Ideal S)
    (hf : Set.LeftInvOn g f (f ⁻¹' I)) :
    I.comap f ≤ I.map g :=
  fun x (hx : f x ∈ I) => hf hx ▸ Ideal.mem_map_of_mem g hx


/-- The `Ideal` version of `Set.image_subset_preimage_of_inverse`. -/
theorem map_le_comap_of_inverse [RingHomClass G S R] (g : G) (I : Ideal R)
    (h : Function.LeftInverse g f) :
    I.map f ≤ I.comap g :=
  map_le_comap_of_inv_on _ _ _ <| h.leftInvOn _


/-- The `Ideal` version of `Set.preimage_subset_image_of_inverse`. -/
theorem comap_le_map_of_inverse (g : G) (I : Ideal S) (h : Function.LeftInverse g f) :
    I.comap f ≤ I.map g :=
  comap_le_map_of_inv_on _ _ _ <| h.leftInvOn _


instance IsPrime.comap [hK : K.IsPrime] : (comap f K).IsPrime :=
                                        /-
                                          R : Type u
                                          S : Type v
                                          F : Type u_1
                                          inst✝⁴ : Semiring R
                                          inst✝³ : Semiring S
                                          inst✝² : FunLike F R S
                                          f : F
                                          I J : Ideal R
                                          K L : Ideal S
                                          G : Type u_2
                                          inst✝¹ : FunLike G S R
                                          inst✝ : RingHomClass F R S
                                          hK : K.IsPrime
                                          x y : R
                                          ⊢ Membership.mem (Ideal.comap f K) (HMul.hMul x y) → Or (Membership.mem (Ideal …
                                        -/
  ⟨comap_ne_top _ hK.1, fun {x y} => by simp only [mem_comap, map_mul]; apply hK.2⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem map_top : map f ⊤ = ⊤ :=
  (eq_top_iff_one _).2 <| subset_span ⟨1, trivial, map_one f⟩


theorem gc_map_comap : GaloisConnection (Ideal.map f) (Ideal.comap f) := fun _ _ =>
  Ideal.map_le_iff_le_comap


@[simp]
theorem comap_id : I.comap (RingHom.id R) = I :=
  Ideal.ext fun _ => Iff.rfl


@[simp]
theorem map_id : I.map (RingHom.id R) = I :=
  (gc_map_comap (RingHom.id R)).l_unique GaloisConnection.id comap_id


theorem comap_comap {T : Type*} [Semiring T] {I : Ideal T} (f : R →+* S) (g : S →+* T) :
    (I.comap g).comap f = I.comap (g.comp f) :=
  rfl


lemma comap_comapₐ {R A B C : Type*} [CommSemiring R] [Semiring A] [Algebra R A] [Semiring B]
    [Algebra R B] [Semiring C] [Algebra R C] {I : Ideal C} (f : A →ₐ[R] B) (g : B →ₐ[R] C) :
    (I.comap g).comap f = I.comap (g.comp f) :=
  I.comap_comap f.toRingHom g.toRingHom


theorem map_map {T : Type*} [Semiring T] {I : Ideal R} (f : R →+* S) (g : S →+* T) :
    (I.map f).map g = I.map (g.comp f) :=
  ((gc_map_comap f).compose (gc_map_comap g)).l_unique (gc_map_comap (g.comp f)) fun _ =>
    comap_comap _ _


lemma map_mapₐ {R A B C : Type*} [CommSemiring R] [Semiring A] [Algebra R A] [Semiring B]
    [Algebra R B] [Semiring C] [Algebra R C] {I : Ideal A} (f : A →ₐ[R] B) (g : B →ₐ[R] C) :
    (I.map f).map g = I.map (g.comp f) :=
  I.map_map f.toRingHom g.toRingHom


theorem map_span (f : F) (s : Set R) : map f (span s) = span (f '' s) := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    s : Set R
    ⊢ Eq (Ideal.map f (Ideal.span s)) (Ideal.span (Set.image (⇑f) s))
  -/
  refine (Submodule.span_eq_of_le _ ?_ ?_).symm
    /-
      case refine_1
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      f : F
      s : Set R
      ⊢ HasSubset.Subset (Set.image (⇑f) s) ↑(Ideal.map f (Ideal.span s))
    -/
  · rintro _ ⟨x, hx, rfl⟩; exact mem_map_of_mem f (subset_span hx)
                           /-
                             🎉 no goals
                           -/
    /-
      case refine_2
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      f : F
      s : Set R
      ⊢ LE.le (Ideal.map f (Ideal.span s)) (Submodule.span S (Set.image (⇑f) s))
    -/
  · rw [map_le_iff_le_comap, span_le, coe_comap, ← Set.image_subset_iff]
    /-
      case refine_2
      R : Type u
      S : Type v
      F : Type u_1
      inst✝³ : Semiring R
      inst✝² : Semiring S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      f : F
      s : Set R
      ⊢ HasSubset.Subset (Set.image (⇑f) s) ↑(Submodule.span S (Set.image (⇑f) s))
    -/
    exact subset_span
    /-
      🎉 no goals
    -/


theorem map_le_of_le_comap : I ≤ K.comap f → I.map f ≤ K :=
  (gc_map_comap f).l_le


theorem le_comap_of_map_le : I.map f ≤ K → I ≤ K.comap f :=
  (gc_map_comap f).le_u


theorem le_comap_map : I ≤ (I.map f).comap f :=
  (gc_map_comap f).le_u_l _


theorem map_comap_le : (K.comap f).map f ≤ K :=
  (gc_map_comap f).l_u_le _


@[simp]
theorem comap_top : (⊤ : Ideal S).comap f = ⊤ :=
  (gc_map_comap f).u_top


@[simp]
theorem comap_eq_top_iff {I : Ideal S} : I.comap f = ⊤ ↔ I = ⊤ :=
  ⟨fun h => I.eq_top_iff_one.mpr (map_one f ▸ mem_comap.mp ((I.comap f).eq_top_iff_one.mp h)),
                /-
                  R : Type u
                  S : Type v
                  F : Type u_1
                  inst✝³ : Semiring R
                  inst✝² : Semiring S
                  inst✝¹ : FunLike F R S
                  f : F
                  inst✝ : RingHomClass F R S
                  I : Ideal S
                  h : Eq I Top.top
                  ⊢ Eq (Ideal.comap f I) Top.top
                -/
    fun h => by rw [h, comap_top]⟩
                /-
                  🎉 no goals
                -/


@[simp]
theorem map_bot : (⊥ : Ideal R).map f = ⊥ :=
  (gc_map_comap f).l_bot


theorem ne_bot_of_map_ne_bot (hI : map f I ≠ ⊥) : I ≠ ⊥ :=
  fun h => hI (Eq.mpr (congrArg (fun I ↦ map f I = ⊥) h) map_bot)


@[simp]
theorem map_comap_map : ((I.map f).comap f).map f = I.map f :=
  (gc_map_comap f).l_u_l_eq_l I


@[simp]
theorem comap_map_comap : ((K.comap f).map f).comap f = K.comap f :=
  (gc_map_comap f).u_l_u_eq_u K


theorem map_sup : (I ⊔ J).map f = I.map f ⊔ J.map f :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).l_sup


theorem comap_inf : comap f (K ⊓ L) = comap f K ⊓ comap f L :=
  rfl


theorem map_iSup (K : ι → Ideal R) : (iSup K).map f = ⨆ i, (K i).map f :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).l_iSup


theorem comap_iInf (K : ι → Ideal S) : (iInf K).comap f = ⨅ i, (K i).comap f :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).u_iInf


theorem map_sSup (s : Set (Ideal R)) : (sSup s).map f = ⨆ I ∈ s, (I : Ideal R).map f :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).l_sSup


theorem comap_sInf (s : Set (Ideal S)) : (sInf s).comap f = ⨅ I ∈ s, (I : Ideal S).comap f :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).u_sInf


theorem comap_sInf' (s : Set (Ideal S)) : (sInf s).comap f = ⨅ I ∈ comap f '' s, I :=
                                    /-
                                      R : Type u
                                      S : Type v
                                      F : Type u_1
                                      inst✝³ : Semiring R
                                      inst✝² : Semiring S
                                      inst✝¹ : FunLike F R S
                                      f : F
                                      inst✝ : RingHomClass F R S
                                      s : Set (Ideal S)
                                      ⊢ Eq (iInf fun I => iInf fun h => Ideal.comap f I) (iInf fun I => iInf fun h = …
                                    -/
  _root_.trans (comap_sInf f s) (by rw [iInf_image])
                                    /-
                                      🎉 no goals
                                    -/


/-- Variant of `Ideal.IsPrime.comap` where ideal is explicit rather than implicit.  -/
theorem comap_isPrime [H : IsPrime K] : IsPrime (comap f K) :=
  H.comap f


theorem map_inf_le : map f (I ⊓ J) ≤ map f I ⊓ map f J :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).monotone_l.map_inf_le _ _


theorem le_comap_sup : comap f K ⊔ comap f L ≤ comap f (K ⊔ L) :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).monotone_u.le_map_sup _ _

-- TODO: Should these be simp lemmas?

theorem _root_.element_smul_restrictScalars {R S M}
    [CommSemiring R] [CommSemiring S] [Algebra R S] [AddCommMonoid M]
    [Module R M] [Module S M] [IsScalarTower R S M] (r : R) (N : Submodule S M) :
    (algebraMap R S r • N).restrictScalars R = r • N.restrictScalars R :=
  SetLike.coe_injective (congrArg (· '' _) (funext (algebraMap_smul S r)))


theorem smul_restrictScalars {R S M} [CommSemiring R] [CommSemiring S]
    [Algebra R S] [AddCommMonoid M] [Module R M] [Module S M]
    [IsScalarTower R S M] (I : Ideal R) (N : Submodule S M) :
    (I.map (algebraMap R S) • N).restrictScalars R = I • N.restrictScalars R := by
  simp_rw [map, Submodule.span_smul_eq, ← Submodule.coe_set_smul,
    Submodule.set_smul_eq_iSup, ← element_smul_restrictScalars, iSup_image]
  /-
    R : Type u_4
    S : Type u_5
    M : Type u_6
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    I : Ideal R
    N : Submodule S M
    ⊢ Eq (Submodule.restrictScalars R (iSup fun b => iSup fun h => HSMul.hSMul ((a …
  -/
  exact map_iSup₂ (Submodule.restrictScalarsLatticeHom R S M) _
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_top_eq_map {R S : Type*} [CommSemiring R] [CommSemiring S] [Algebra R S]
    (I : Ideal R) : I • (⊤ : Submodule R S) = (I.map (algebraMap R S)).restrictScalars R :=
  Eq.trans (smul_restrictScalars I (⊤ : Ideal S)).symm <|
    congrArg _ <| Eq.trans (Ideal.smul_eq_mul _ _) (Ideal.mul_top _)


@[simp]
theorem coe_restrictScalars {R S : Type*} [Semiring R] [Semiring S] [Module R S]
    [IsScalarTower R S S] (I : Ideal S) : (I.restrictScalars R : Set S) = ↑I :=
  rfl


/-- The smallest `S`-submodule that contains all `x ∈ I * y ∈ J`
is also the smallest `R`-submodule that does so. -/
@[simp]
theorem restrictScalars_mul {R S : Type*} [Semiring R] [Semiring S] [Module R S]
    [IsScalarTower R S S] (I J : Ideal S) :
    (I * J).restrictScalars R = I.restrictScalars R * J.restrictScalars R :=
  rfl


theorem map_comap_of_surjective (I : Ideal S) : map f (comap f I) = I :=
  le_antisymm (map_le_iff_le_comap.2 le_rfl) fun s hsi =>
    let ⟨r, hfrs⟩ := hf s
    hfrs ▸ (mem_map_of_mem f <| show f r ∈ I from hfrs.symm ▸ hsi)


/-- `map` and `comap` are adjoint, and the composition `map f ∘ comap f` is the
  identity -/
def giMapComap : GaloisInsertion (map f) (comap f) :=
  GaloisInsertion.monotoneIntro (gc_map_comap f).monotone_u (gc_map_comap f).monotone_l
    (fun _ => le_comap_map) (map_comap_of_surjective _ hf)


theorem map_surjective_of_surjective : Surjective (map f) :=
  (giMapComap f hf).l_surjective


theorem comap_injective_of_surjective : Injective (comap f) :=
  (giMapComap f hf).u_injective


theorem map_sup_comap_of_surjective (I J : Ideal S) : (I.comap f ⊔ J.comap f).map f = I ⊔ J :=
  (giMapComap f hf).l_sup_u _ _


theorem map_iSup_comap_of_surjective (K : ι → Ideal S) : (⨆ i, (K i).comap f).map f = iSup K :=
  (giMapComap f hf).l_iSup_u _


theorem map_inf_comap_of_surjective (I J : Ideal S) : (I.comap f ⊓ J.comap f).map f = I ⊓ J :=
  (giMapComap f hf).l_inf_u _ _


theorem map_iInf_comap_of_surjective (K : ι → Ideal S) : (⨅ i, (K i).comap f).map f = iInf K :=
  (giMapComap f hf).l_iInf_u _


theorem mem_image_of_mem_map_of_surjective {I : Ideal R} {y} (H : y ∈ map f I) : y ∈ f '' I :=
  Submodule.span_induction (hx := H) (fun _ => id) ⟨0, I.zero_mem, map_zero f⟩
    (fun _ _ _ _ ⟨x1, hx1i, hxy1⟩ ⟨x2, hx2i, hxy2⟩ =>
      ⟨x1 + x2, I.add_mem hx1i hx2i, hxy1 ▸ hxy2 ▸ map_add f _ _⟩)
    fun c _ _ ⟨x, hxi, hxy⟩ =>
    let ⟨d, hdc⟩ := hf c
    ⟨d * x, I.mul_mem_left _ hxi, hdc ▸ hxy ▸ map_mul f _ _⟩


theorem mem_map_iff_of_surjective {I : Ideal R} {y} : y ∈ map f I ↔ ∃ x, x ∈ I ∧ f x = y :=
  ⟨fun h => (Set.mem_image _ _ _).2 (mem_image_of_mem_map_of_surjective f hf h), fun ⟨_, hx⟩ =>
    hx.right ▸ mem_map_of_mem f hx.left⟩


theorem le_map_of_comap_le_of_surjective : comap f K ≤ I → K ≤ map f I := fun h =>
  map_comap_of_surjective f hf K ▸ map_mono h


theorem map_comap_eq_self_of_equiv {E : Type*} [EquivLike E R S] [RingEquivClass E R S] (e : E)
    (I : Ideal S) : map e (comap e I) = I :=
  I.map_comap_of_surjective e (EquivLike.surjective e)


theorem map_eq_submodule_map (f : R →+* S) [h : RingHomSurjective f] (I : Ideal R) :
    I.map f = Submodule.map f.toSemilinearMap I :=
  Submodule.ext fun _ => mem_map_iff_of_surjective f h.1


open Function in
theorem IsMaximal.comap_piEvalRingHom {ι : Type*} {R : ι → Type*} [∀ i, Semiring (R i)]
    {i : ι} {I : Ideal (R i)} (h : I.IsMaximal) : (I.comap <| Pi.evalRingHom R i).IsMaximal := by
  /-
    ι : Type u_4
    R : ι → Type u_5
    inst✝ : (i : ι) → Semiring (R i)
    i : ι
    I : Ideal (R i)
    h : I.IsMaximal
    ⊢ (Ideal.comap (Pi.evalRingHom R i) I).IsMaximal
  -/
  refine isMaximal_iff.mpr ⟨I.ne_top_iff_one.mp h.ne_top, fun J x le hxI hxJ ↦ ?_⟩
  /-
    ι : Type u_4
    R : ι → Type u_5
    inst✝ : (i : ι) → Semiring (R i)
    i : ι
    I : Ideal (R i)
    h : I.IsMaximal
    J : Ideal ((i : ι) → R i)
    x : (i : ι) → R i
    le : LE.le (Ideal.comap (Pi.evalRingHom R i) I) J
    hxI : Not (Membership.mem (Ideal.comap (Pi.evalRingHom R i) I) x)
    hxJ : Membership.mem J x
    ⊢ Membership.mem J 1
  -/
  have ⟨r, y, hy, eq⟩ := h.exists_inv hxI
  classical
  convert J.add_mem (J.mul_mem_left (update 0 i r) hxJ)
    (b := update 1 i y) (le <| by apply update_self i y 1 ▸ hy)
  ext j
  obtain rfl | ne := eq_or_ne j i
  · simpa [eq_comm] using eq
  · simp [update_of_ne ne]


theorem comap_bot_le_of_injective (hf : Function.Injective f) : comap f ⊥ ≤ I := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    f : F
    I : Ideal R
    inst✝ : RingHomClass F R S
    hf : Function.Injective ⇑f
    ⊢ LE.le (Ideal.comap f Bot.bot) I
  -/
  refine le_trans (fun x hx => ?_) bot_le
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    f : F
    I : Ideal R
    inst✝ : RingHomClass F R S
    hf : Function.Injective ⇑f
    x : R
    hx : Membership.mem (Ideal.comap f Bot.bot) x
    ⊢ Membership.mem Bot.bot x
  -/
  rw [mem_comap, Submodule.mem_bot, ← map_zero f] at hx
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    f : F
    I : Ideal R
    inst✝ : RingHomClass F R S
    hf : Function.Injective ⇑f
    x : R
    hx : Eq (f x) (f 0)
    ⊢ Membership.mem Bot.bot x
  -/
  exact Eq.symm (hf hx) ▸ Submodule.zero_mem ⊥
  /-
    🎉 no goals
  -/


theorem comap_bot_of_injective (hf : Function.Injective f) : Ideal.comap f ⊥ = ⊥ :=
  le_bot_iff.mp (Ideal.comap_bot_le_of_injective f hf)


/-- If `f : R ≃+* S` is a ring isomorphism and `I : Ideal R`, then `map f.symm (map f I) = I`. -/
@[simp]
theorem map_of_equiv {I : Ideal R} (f : R ≃+* S) :
    (I.map (f : R →+* S)).map (f.symm : S →+* R) = I := by
  rw [← RingEquiv.toRingHom_eq_coe, ← RingEquiv.toRingHom_eq_coe, map_map,
    RingEquiv.toRingHom_eq_coe, RingEquiv.toRingHom_eq_coe, RingEquiv.symm_comp, map_id]


/-- If `f : R ≃+* S` is a ring isomorphism and `I : Ideal R`,
  then `comap f (comap f.symm I) = I`. -/
@[simp]
theorem comap_of_equiv {I : Ideal R} (f : R ≃+* S) :
    (I.comap (f.symm : S →+* R)).comap (f : R →+* S) = I := by
  rw [← RingEquiv.toRingHom_eq_coe, ← RingEquiv.toRingHom_eq_coe, comap_comap,
    RingEquiv.toRingHom_eq_coe, RingEquiv.toRingHom_eq_coe, RingEquiv.symm_comp, comap_id]


/-- If `f : R ≃+* S` is a ring isomorphism and `I : Ideal R`, then `map f I = comap f.symm I`. -/
theorem map_comap_of_equiv {I : Ideal R} (f : R ≃+* S) : I.map (f : R →+* S) = I.comap f.symm :=
  le_antisymm (Ideal.map_le_comap_of_inverse _ _ _ (Equiv.left_inv' _))
    (Ideal.comap_le_map_of_inverse _ _ _ (Equiv.right_inv' _))


/-- If `f : R ≃+* S` is a ring isomorphism and `I : Ideal R`, then `comap f.symm I = map f I`. -/
@[simp]
theorem comap_symm {I : Ideal R} (f : R ≃+* S) : I.comap f.symm = I.map f :=
  (map_comap_of_equiv f).symm


/-- If `f : R ≃+* S` is a ring isomorphism and `I : Ideal R`, then `map f.symm I = comap f I`. -/
@[simp]

theorem map_symm {I : Ideal S} (f : R ≃+* S) : I.map f.symm = I.comap f :=
  map_comap_of_equiv (RingEquiv.symm f)


@[simp]
theorem symm_apply_mem_of_equiv_iff {I : Ideal R} {f : R ≃+* S} {y : S} :
    f.symm y ∈ I ↔ y ∈ I.map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    f : RingEquiv R S
    y : S
    ⊢ Iff (Membership.mem I (f.symm y)) (Membership.mem (Ideal.map f I) y)
  -/
  rw [← comap_symm, mem_comap]
  /-
    🎉 no goals
  -/


@[simp]
theorem apply_mem_of_equiv_iff {I : Ideal R} {f : R ≃+* S} {x : R} :
    f x ∈ I.map f ↔ x ∈ I := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    I : Ideal R
    f : RingEquiv R S
    x : R
    ⊢ Iff (Membership.mem (Ideal.map f I) (f x)) (Membership.mem I x)
  -/
  rw [← comap_symm, Ideal.mem_comap, f.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem mem_map_of_equiv {E : Type*} [EquivLike E R S] [RingEquivClass E R S] (e : E)
    {I : Ideal R} (y : S) : y ∈ map e I ↔ ∃ x ∈ I, e x = y := by
  /-
    R : Type u
    S : Type v
    inst✝³ : Semiring R
    inst✝² : Semiring S
    E : Type u_4
    inst✝¹ : EquivLike E R S
    inst✝ : RingEquivClass E R S
    e : E
    I : Ideal R
    y : S
    ⊢ Iff (Membership.mem (Ideal.map e I) y) (Exists fun x => And (Membership.mem  …
  -/
  constructor
    /-
      case mp
      R : Type u
      S : Type v
      inst✝³ : Semiring R
      inst✝² : Semiring S
      E : Type u_4
      inst✝¹ : EquivLike E R S
      inst✝ : RingEquivClass E R S
      e : E
      I : Ideal R
      y : S
      ⊢ Membership.mem (Ideal.map e I) y → Exists fun x => And (Membership.mem I x)  …
    -/
  · intro h
    /-
      case mp
      R : Type u
      S : Type v
      inst✝³ : Semiring R
      inst✝² : Semiring S
      E : Type u_4
      inst✝¹ : EquivLike E R S
      inst✝ : RingEquivClass E R S
      e : E
      I : Ideal R
      y : S
      h : Membership.mem (Ideal.map e I) y
      ⊢ Exists fun x => And (Membership.mem I x) (Eq (e x) y)
    -/
    simp_rw [show map e I = _ from map_comap_of_equiv (e : R ≃+* S)] at h
    /-
      case mp
      R : Type u
      S : Type v
      inst✝³ : Semiring R
      inst✝² : Semiring S
      E : Type u_4
      inst✝¹ : EquivLike E R S
      inst✝ : RingEquivClass E R S
      e : E
      I : Ideal R
      y : S
      h : Membership.mem (Ideal.comap (↑e).symm I) y
      ⊢ Exists fun x => And (Membership.mem I x) (Eq (e x) y)
    -/
    exact ⟨(e : R ≃+* S).symm y, h, (e : R ≃+* S).apply_symm_apply y⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝³ : Semiring R
      inst✝² : Semiring S
      E : Type u_4
      inst✝¹ : EquivLike E R S
      inst✝ : RingEquivClass E R S
      e : E
      I : Ideal R
      y : S
      ⊢ (Exists fun x => And (Membership.mem I x) (Eq (e x) y)) → Membership.mem (Id …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case mpr.intro.intro
      R : Type u
      S : Type v
      inst✝³ : Semiring R
      inst✝² : Semiring S
      E : Type u_4
      inst✝¹ : EquivLike E R S
      inst✝ : RingEquivClass E R S
      e : E
      I : Ideal R
      x : R
      hx : Membership.mem I x
      ⊢ Membership.mem (Ideal.map e I) (e x)
    -/
    exact mem_map_of_mem e hx
    /-
      🎉 no goals
    -/


/-- Special case of the correspondence theorem for isomorphic rings -/
def relIsoOfBijective : Ideal S ≃o Ideal R where
  toFun := comap f
  invFun := map f
  left_inv := map_comap_of_surjective _ hf.2
  right_inv J :=
    le_antisymm
      (fun _ h ↦ have ⟨y, hy, eq⟩ := (mem_map_iff_of_surjective _ hf.2).mp h; hf.1 eq ▸ hy)
      le_comap_map
  map_rel_iff' {_ _} := by
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝⁴ : Semiring R
      inst✝³ : Semiring S
      inst✝² : FunLike F R S
      f : F
      I✝ J : Ideal R
      K✝ L : Ideal S
      G : Type u_2
      inst✝¹ : FunLike G S R
      inst✝ : RingHomClass F R S
      ι : Sort u_3
      hf : Function.Bijective ⇑f
      I : Ideal R
      K x✝¹ x✝ : Ideal S
      ⊢ Iff (LE.le ({ toFun := Ideal.comap f, invFun := Ideal.map f, left_inv := ⋯,  …
    -/
    refine ⟨fun h ↦ ?_, comap_mono⟩
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝⁴ : Semiring R
      inst✝³ : Semiring S
      inst✝² : FunLike F R S
      f : F
      I✝ J : Ideal R
      K✝ L : Ideal S
      G : Type u_2
      inst✝¹ : FunLike G S R
      inst✝ : RingHomClass F R S
      ι : Sort u_3
      hf : Function.Bijective ⇑f
      I : Ideal R
      K x✝¹ x✝ : Ideal S
      h : LE.le ({ toFun := Ideal.comap f, invFun := Ideal.map f, left_inv := ⋯, rig …
      ⊢ LE.le x✝¹ x✝
    -/
    have := map_mono (f := f) h
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝⁴ : Semiring R
      inst✝³ : Semiring S
      inst✝² : FunLike F R S
      f : F
      I✝ J : Ideal R
      K✝ L : Ideal S
      G : Type u_2
      inst✝¹ : FunLike G S R
      inst✝ : RingHomClass F R S
      ι : Sort u_3
      hf : Function.Bijective ⇑f
      I : Ideal R
      K x✝¹ x✝ : Ideal S
      h : LE.le ({ toFun := Ideal.comap f, invFun := Ideal.map f, left_inv := ⋯, rig …
      this : LE.le (Ideal.map f ({ toFun := Ideal.comap f, invFun := Ideal.map f, le …
      ⊢ LE.le x✝¹ x✝
    -/
    simpa only [Equiv.coe_fn_mk, map_comap_of_surjective f hf.2] using this
    /-
      🎉 no goals
    -/


theorem comap_le_iff_le_map : comap f K ≤ I ↔ K ≤ map f I :=
  ⟨fun h => le_map_of_comap_le_of_surjective f hf.right h, fun h =>
    (relIsoOfBijective f hf).right_inv I ▸ comap_mono h⟩


lemma comap_map_of_bijective : (I.map f).comap f = I :=
  le_antisymm ((comap_le_iff_le_map f hf).mpr fun _ ↦ id) le_comap_map


theorem isMaximal_map_iff_of_bijective : IsMaximal (map f I) ↔ IsMaximal I := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    f : F
    inst✝ : RingHomClass F R S
    hf : Function.Bijective ⇑f
    I : Ideal R
    ⊢ Iff (Ideal.map f I).IsMaximal I.IsMaximal
  -/
  simpa only [isMaximal_def] using (relIsoOfBijective _ hf).symm.isCoatom_iff _
  /-
    🎉 no goals
  -/


theorem isMaximal_comap_iff_of_bijective : IsMaximal (comap f K) ↔ IsMaximal K := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    f : F
    inst✝ : RingHomClass F R S
    hf : Function.Bijective ⇑f
    K : Ideal S
    ⊢ Iff (Ideal.comap f K).IsMaximal K.IsMaximal
  -/
  simpa only [isMaximal_def] using (relIsoOfBijective _ hf).isCoatom_iff _
  /-
    🎉 no goals
  -/


alias ⟨_, IsMaximal.map_bijective⟩ := isMaximal_map_iff_of_bijective

alias ⟨_, IsMaximal.comap_bijective⟩ := isMaximal_comap_iff_of_bijective


/-- A ring isomorphism sends a maximal ideal to a maximal ideal. -/
instance map_isMaximal_of_equiv {E : Type*} [EquivLike E R S] [RingEquivClass E R S] (e : E)
    {p : Ideal R} [hp : p.IsMaximal] : (map e p).IsMaximal :=
  hp.map_bijective e (EquivLike.bijective e)


theorem isMaximal_iff_of_bijective : (⊥ : Ideal R).IsMaximal ↔ (⊥ : Ideal S).IsMaximal :=
  ⟨fun h ↦ map_bot (f := f) ▸ h.map_bijective f hf, fun h ↦ have e := RingEquiv.ofBijective f hf
    map_bot (f := e.symm) ▸ h.map_bijective _ e.symm.bijective⟩


@[deprecated (since := "2024-12-07")] alias map.isMaximal := IsMaximal.map_bijective

@[deprecated (since := "2024-12-07")] alias comap.isMaximal := IsMaximal.comap_bijective

@[deprecated (since := "2024-12-07")] alias RingEquiv.bot_maximal_iff := isMaximal_iff_of_bijective


theorem comap_map_of_surjective (hf : Function.Surjective f) (I : Ideal R) :
    comap f (map f I) = I ⊔ comap f ⊥ :=
  le_antisymm
    (fun r h =>
      let ⟨s, hsi, hfsr⟩ := mem_image_of_mem_map_of_surjective f hf h
      Submodule.mem_sup.2
                                                      /-
                                                        R : Type u
                                                        S : Type v
                                                        F : Type u_1
                                                        inst✝³ : Ring R
                                                        inst✝² : Ring S
                                                        inst✝¹ : FunLike F R S
                                                        inst✝ : RingHomClass F R S
                                                        f : F
                                                        hf : Function.Surjective ⇑f
                                                        I : Ideal R
                                                        r : R
                                                        h : Membership.mem (Ideal.map f I) (f r)
                                                        s : R
                                                        hsi : Membership.mem (↑I) s
                                                        hfsr : Eq (f s) (f r)
                                                        ⊢ Eq (f (HSub.hSub r s)) 0
                                                      -/
        ⟨s, hsi, r - s, (Submodule.mem_bot S).2 <| by rw [map_sub, hfsr, sub_self],
                                                      /-
                                                        🎉 no goals
                                                      -/
          add_sub_cancel s r⟩)
    (sup_le (map_le_iff_le_comap.1 le_rfl) (comap_mono bot_le))


/-- Correspondence theorem -/
def relIsoOfSurjective (hf : Function.Surjective f) :
    Ideal S ≃o { p : Ideal R // comap f ⊥ ≤ p } where
  toFun J := ⟨comap f J, comap_mono bot_le⟩
  invFun I := map f I.1
  left_inv J := map_comap_of_surjective f hf J
  right_inv I :=
    Subtype.eq <|
      show comap f (map f I.1) = I.1 from
        (comap_map_of_surjective f hf I).symm ▸ le_antisymm (sup_le le_rfl I.2) le_sup_left
  map_rel_iff' {I1 I2} :=
    ⟨fun H => map_comap_of_surjective f hf I1 ▸ map_comap_of_surjective f hf I2 ▸ map_mono H,
      comap_mono⟩


/-- The map on ideals induced by a surjective map preserves inclusion. -/
def orderEmbeddingOfSurjective (hf : Function.Surjective f) : Ideal S ↪o Ideal R :=
  (relIsoOfSurjective f hf).toRelEmbedding.trans (Subtype.relEmbedding (fun x y => x ≤ y) _)


theorem map_eq_top_or_isMaximal_of_surjective (hf : Function.Surjective f) {I : Ideal R}
    (H : IsMaximal I) : map f I = ⊤ ∨ IsMaximal (map f I) := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Ring R
    inst✝² : Ring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    I : Ideal R
    H : I.IsMaximal
    ⊢ Or (Eq (Ideal.map f I) Top.top) (Ideal.map f I).IsMaximal
  -/
  refine or_iff_not_imp_left.2 fun ne_top => ⟨⟨fun h => ne_top h, fun J hJ => ?_⟩⟩
  · refine
      (relIsoOfSurjective f hf).injective
        (Subtype.ext_iff.2 (Eq.trans (H.1.2 (comap f J) (lt_of_le_of_ne ?_ ?_)) comap_top.symm))
      /-
        case refine_1
        R : Type u
        S : Type v
        F : Type u_1
        inst✝³ : Ring R
        inst✝² : Ring S
        inst✝¹ : FunLike F R S
        inst✝ : RingHomClass F R S
        f : F
        hf : Function.Surjective ⇑f
        I : Ideal R
        H : I.IsMaximal
        ne_top : Not (Eq (Ideal.map f I) Top.top)
        J : Ideal S
        hJ : LT.lt (Ideal.map f I) J
        ⊢ LE.le I (Ideal.comap f J)
      -/
    · exact map_le_iff_le_comap.1 (le_of_lt hJ)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u
        S : Type v
        F : Type u_1
        inst✝³ : Ring R
        inst✝² : Ring S
        inst✝¹ : FunLike F R S
        inst✝ : RingHomClass F R S
        f : F
        hf : Function.Surjective ⇑f
        I : Ideal R
        H : I.IsMaximal
        ne_top : Not (Eq (Ideal.map f I) Top.top)
        J : Ideal S
        hJ : LT.lt (Ideal.map f I) J
        ⊢ Ne I (Ideal.comap f J)
      -/
    · exact fun h => hJ.right (le_map_of_comap_le_of_surjective f hf (le_of_eq h.symm))
      /-
        🎉 no goals
      -/


theorem comap_isMaximal_of_surjective (hf : Function.Surjective f) {K : Ideal S} [H : IsMaximal K] :
    IsMaximal (comap f K) := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Ring R
    inst✝² : Ring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    K : Ideal S
    H : K.IsMaximal
    ⊢ (Ideal.comap f K).IsMaximal
  -/
  refine ⟨⟨comap_ne_top _ H.1.1, fun J hJ => ?_⟩⟩
  suffices map f J = ⊤ by
    have := congr_arg (comap f) this
    rw [comap_top, comap_map_of_surjective _ hf, eq_top_iff] at this
    rw [eq_top_iff]
    exact le_trans this (sup_le (le_of_eq rfl) (le_trans (comap_mono bot_le) (le_of_lt hJ)))
  refine
    H.1.2 (map f J)
      (lt_of_le_of_ne (le_map_of_comap_le_of_surjective _ hf (le_of_lt hJ)) fun h =>
        ne_of_lt hJ (_root_.trans (congr_arg (comap f) h) ?_))
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Ring R
    inst✝² : Ring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    K : Ideal S
    H : K.IsMaximal
    J : Ideal R
    hJ : LT.lt (Ideal.comap f K) J
    h : Eq K (Ideal.map f J)
    ⊢ Eq (Ideal.comap f (Ideal.map f J)) J
  -/
  rw [comap_map_of_surjective _ hf, sup_eq_left]
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Ring R
    inst✝² : Ring S
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    K : Ideal S
    H : K.IsMaximal
    J : Ideal R
    hJ : LT.lt (Ideal.comap f K) J
    h : Eq K (Ideal.map f J)
    ⊢ LE.le (Ideal.comap f Bot.bot) J
  -/
  exact le_trans (comap_mono bot_le) (le_of_lt hJ)
  /-
    🎉 no goals
  -/


/-- The pullback of a maximal ideal under a ring isomorphism is a maximal ideal. -/
instance comap_isMaximal_of_equiv {E : Type*} [EquivLike E R S] [RingEquivClass E R S] (e : E)
    {p : Ideal S} [p.IsMaximal] : (comap e p).IsMaximal :=
  comap_isMaximal_of_surjective e (EquivLike.surjective e)


theorem comap_le_comap_iff_of_surjective (hf : Function.Surjective f) (I J : Ideal S) :
    comap f I ≤ comap f J ↔ I ≤ J :=
  ⟨fun h => (map_comap_of_surjective f hf I).symm.le.trans (map_le_of_le_comap h), fun h =>
    le_comap_of_map_le ((map_comap_of_surjective f hf I).le.trans h)⟩


theorem map_mul {R} [Semiring R] [FunLike F R S] [RingHomClass F R S] (f : F) (I J : Ideal R) :
    map f (I * J) = map f I * map f J :=
  le_antisymm
    (map_le_iff_le_comap.2 <|
      mul_le.2 fun r hri s hsj =>
        show (f (r * s)) ∈ map f I * map f J by
          /-
            S : Type v
            F : Type u_1
            inst✝³ : CommSemiring S
            R : Type u_2
            inst✝² : Semiring R
            inst✝¹ : FunLike F R S
            inst✝ : RingHomClass F R S
            f : F
            I J : Ideal R
            r : R
            hri : Membership.mem I r
            s : R
            hsj : Membership.mem J s
            ⊢ Membership.mem (HMul.hMul (Ideal.map f I) (Ideal.map f J)) (f (HMul.hMul r s))
          -/
          rw [_root_.map_mul]; exact mul_mem_mul (mem_map_of_mem f hri) (mem_map_of_mem f hsj))
                               /-
                                 🎉 no goals
                               -/
    (span_mul_span (↑f '' ↑I) (↑f '' ↑J) ▸ (span_le.2 <|
      Set.iUnion₂_subset fun _ ⟨r, hri, hfri⟩ =>
        Set.iUnion₂_subset fun _ ⟨s, hsj, hfsj⟩ =>
          Set.singleton_subset_iff.2 <|
                             /-
                               S : Type v
                               F : Type u_1
                               inst✝³ : CommSemiring S
                               R : Type u_2
                               inst✝² : Semiring R
                               inst✝¹ : FunLike F R S
                               inst✝ : RingHomClass F R S
                               f : F
                               I J : Ideal R
                               x✝³ : S
                               x✝² : Membership.mem (Set.image ⇑f ↑I) x✝³
                               r : R
                               hri : Membership.mem (↑I) r
                               hfri : Eq (f r) x✝³
                               x✝¹ : S
                               x✝ : Membership.mem (Set.image ⇑f ↑J) x✝¹
                               s : R
                               hsj : Membership.mem (↑J) s
                               hfsj : Eq (f s) x✝¹
                               ⊢ Membership.mem (↑(Ideal.map f (HMul.hMul I J))) (HMul.hMul (f r) (f s))
                             -/
            hfri ▸ hfsj ▸ by rw [← _root_.map_mul]; exact mem_map_of_mem f (mul_mem_mul hri hsj)))
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The pushforward `Ideal.map` as a (semi)ring homomorphism. -/
@[simps]
def mapHom : Ideal R →+* Ideal S where
  toFun := map f
  map_mul' := Ideal.map_mul f
                 /-
                   R : Type u
                   S : Type v
                   F : Type u_1
                   inst✝² : CommSemiring R
                   inst✝¹ : CommSemiring S
                   inst✝ : FunLike F R S
                   rc : RingHomClass F R S
                   f : F
                   I J : Ideal R
                   K L : Ideal S
                   ⊢ Eq (Ideal.map f 1) 1
                 -/
  map_one' := by simp only [one_eq_top]; exact Ideal.map_top f
                                         /-
                                           🎉 no goals
                                         -/
  map_add' I J := Ideal.map_sup f I J
  map_zero' := Ideal.map_bot


protected theorem map_pow (n : ℕ) : map f (I ^ n) = map f I ^ n :=
  map_pow (mapHom f) I n


theorem comap_radical : comap f (radical K) = radical (comap f K) := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    K : Ideal S
    ⊢ Eq (Ideal.comap f K.radical) (Ideal.comap f K).radical
  -/
  ext
  /-
    case h
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    K : Ideal S
    x✝ : R
    ⊢ Iff (Membership.mem (Ideal.comap f K.radical) x✝) (Membership.mem (Ideal.com …
  -/
  simp [radical]
  /-
    🎉 no goals
  -/


theorem IsRadical.comap (hK : K.IsRadical) : (comap f K).IsRadical := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    K : Ideal S
    hK : K.IsRadical
    ⊢ (Ideal.comap f K).IsRadical
  -/
  rw [← hK.radical, comap_radical]
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    K : Ideal S
    hK : K.IsRadical
    ⊢ (Ideal.comap f K).radical.IsRadical
  -/
  apply radical_isRadical
  /-
    🎉 no goals
  -/


theorem map_radical_le : map f (radical I) ≤ radical (map f I) :=
  map_le_iff_le_comap.2 fun r ⟨n, hrni⟩ => ⟨n, map_pow f r n ▸ mem_map_of_mem f hrni⟩


theorem le_comap_mul : comap f K * comap f L ≤ comap f (K * L) :=
  map_le_iff_le_comap.1 <|
    (map_mul f (comap f K) (comap f L)).symm ▸
      mul_mono (map_le_iff_le_comap.2 <| le_rfl) (map_le_iff_le_comap.2 <| le_rfl)


theorem le_comap_pow (n : ℕ) : K.comap f ^ n ≤ (K ^ n).comap f := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    K : Ideal S
    n : Nat
    ⊢ LE.le (HPow.hPow (Ideal.comap f K) n) (Ideal.comap f (HPow.hPow K n))
  -/
  induction' n with n n_ih
    /-
      case zero
      R : Type u
      S : Type v
      F : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      K : Ideal S
      ⊢ LE.le (HPow.hPow (Ideal.comap f K) 0) (Ideal.comap f (HPow.hPow K 0))
    -/
  · rw [pow_zero, pow_zero, Ideal.one_eq_top, Ideal.one_eq_top]
    /-
      case zero
      R : Type u
      S : Type v
      F : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      K : Ideal S
      ⊢ LE.le Top.top (Ideal.comap f Top.top)
    -/
    exact rfl.le
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      S : Type v
      F : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      K : Ideal S
      n : Nat
      n_ih : LE.le (HPow.hPow (Ideal.comap f K) n) (Ideal.comap f (HPow.hPow K n))
      ⊢ LE.le (HPow.hPow (Ideal.comap f K) (HAdd.hAdd n 1)) (Ideal.comap f (HPow.hPo …
    -/
  · rw [pow_succ, pow_succ]
    /-
      case succ
      R : Type u
      S : Type v
      F : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      K : Ideal S
      n : Nat
      n_ih : LE.le (HPow.hPow (Ideal.comap f K) n) (Ideal.comap f (HPow.hPow K n))
      ⊢ LE.le (HMul.hMul (HPow.hPow (Ideal.comap f K) n) (Ideal.comap f K)) (Ideal.c …
    -/
    exact (Ideal.mul_mono_left n_ih).trans (Ideal.le_comap_mul f)
    /-
      🎉 no goals
    -/


/-- Kernel of a ring homomorphism as an ideal of the domain. -/
def ker : Ideal R :=
  Ideal.comap f ⊥


variable {f} in
/-- An element is in the kernel if and only if it maps to zero. -/
                                                        /-
                                                          R : Type u
                                                          S : Type v
                                                          F : Type u_1
                                                          inst✝² : Semiring R
                                                          inst✝¹ : Semiring S
                                                          inst✝ : FunLike F R S
                                                          rcf : RingHomClass F R S
                                                          f : F
                                                          r : R
                                                          ⊢ Iff (Membership.mem (RingHom.ker f) r) (Eq (f r) 0)
                                                        -/
@[simp] theorem mem_ker {r} : r ∈ ker f ↔ f r = 0 := by rw [ker, Ideal.mem_comap, Submodule.mem_bot]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem ker_eq : (ker f : Set R) = Set.preimage f {0} :=
  rfl


theorem ker_eq_comap_bot (f : F) : ker f = Ideal.comap f ⊥ :=
  rfl


theorem comap_ker (f : S →+* R) (g : T →+* S) : f.ker.comap g = ker (f.comp g) := by
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    inst✝ : Semiring T
    f : RingHom S R
    g : RingHom T S
    ⊢ Eq (Ideal.comap g (RingHom.ker f)) (RingHom.ker (f.comp g))
  -/
  rw [RingHom.ker_eq_comap_bot, Ideal.comap_comap, RingHom.ker_eq_comap_bot]
  /-
    🎉 no goals
  -/


/-- If the target is not the zero ring, then one is not in the kernel. -/
theorem not_one_mem_ker [Nontrivial S] (f : F) : (1 : R) ∉ ker f := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    rcf : RingHomClass F R S
    inst✝ : Nontrivial S
    f : F
    ⊢ Not (Membership.mem (RingHom.ker f) 1)
  -/
  rw [mem_ker, map_one]
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : FunLike F R S
    rcf : RingHomClass F R S
    inst✝ : Nontrivial S
    f : F
    ⊢ Not (Eq 1 0)
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


theorem ker_ne_top [Nontrivial S] (f : F) : ker f ≠ ⊤ :=
  (Ideal.ne_top_iff_one _).mpr <| not_one_mem_ker f


lemma _root_.Pi.ker_ringHom {ι : Type*} {R : ι → Type*} [∀ i, Semiring (R i)]
    (φ : ∀ i, S →+* R i) : ker (Pi.ringHom φ) = ⨅ i, ker (φ i) := by
  /-
    S : Type v
    inst✝¹ : Semiring S
    ι : Type u_3
    R : ι → Type u_4
    inst✝ : (i : ι) → Semiring (R i)
    φ : (i : ι) → RingHom S (R i)
    ⊢ Eq (RingHom.ker (Pi.ringHom φ)) (iInf fun i => RingHom.ker (φ i))
  -/
  ext x
  /-
    case h
    S : Type v
    inst✝¹ : Semiring S
    ι : Type u_3
    R : ι → Type u_4
    inst✝ : (i : ι) → Semiring (R i)
    φ : (i : ι) → RingHom S (R i)
    x : S
    ⊢ Iff (Membership.mem (RingHom.ker (Pi.ringHom φ)) x) (Membership.mem (iInf fu …
  -/
  simp [mem_ker, Ideal.mem_iInf, funext_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_rangeSRestrict (f : R →+* S) : ker f.rangeSRestrict = ker f :=
  Ideal.ext fun _ ↦ Subtype.ext_iff


theorem injective_iff_ker_eq_bot : Function.Injective f ↔ ker f = ⊥ := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : Ring R
    inst✝¹ : Semiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    ⊢ Iff (Function.Injective ⇑f) (Eq (RingHom.ker f) Bot.bot)
  -/
  rw [SetLike.ext'_iff, ker_eq, Set.ext_iff]
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : Ring R
    inst✝¹ : Semiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    ⊢ Iff (Function.Injective ⇑f) (∀ (x : R), Iff (Membership.mem (Set.preimage (⇑ …
  -/
  exact injective_iff_map_eq_zero' f
  /-
    🎉 no goals
  -/


theorem ker_eq_bot_iff_eq_zero : ker f = ⊥ ↔ ∀ x, f x = 0 → x = 0 := by
  /-
    R : Type u
    S : Type v
    F : Type u_1
    inst✝² : Ring R
    inst✝¹ : Semiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    ⊢ Iff (Eq (RingHom.ker f) Bot.bot) (∀ (x : R), Eq (f x) 0 → Eq x 0)
  -/
  rw [← injective_iff_map_eq_zero f, injective_iff_ker_eq_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_coe_equiv (f : R ≃+* S) : ker (f : R →+* S) = ⊥ := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    inst✝ : Semiring S
    f : RingEquiv R S
    ⊢ Eq (RingHom.ker ↑f) Bot.bot
  -/
  simpa only [← injective_iff_ker_eq_bot] using EquivLike.injective f
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_equiv {F' : Type*} [EquivLike F' R S] [RingEquivClass F' R S] (f : F') : ker f = ⊥ := by
  /-
    R : Type u
    S : Type v
    inst✝³ : Ring R
    inst✝² : Semiring S
    F' : Type u_2
    inst✝¹ : EquivLike F' R S
    inst✝ : RingEquivClass F' R S
    f : F'
    ⊢ Eq (RingHom.ker f) Bot.bot
  -/
  simpa only [← injective_iff_ker_eq_bot] using EquivLike.injective f
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  R : Type u
                                                                  S : Type v
                                                                  F : Type u_1
                                                                  inst✝² : Ring R
                                                                  inst✝¹ : Ring S
                                                                  inst✝ : FunLike F R S
                                                                  rc : RingHomClass F R S
                                                                  f : F
                                                                  x y : R
                                                                  ⊢ Iff (Membership.mem (RingHom.ker f) (HSub.hSub x y)) (Eq (f x) (f y))
                                                                -/
theorem sub_mem_ker_iff {x y} : x - y ∈ ker f ↔ f x = f y := by rw [mem_ker, map_sub, sub_eq_zero]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem ker_rangeRestrict (f : R →+* S) : ker f.rangeRestrict = ker f :=
  Ideal.ext fun _ ↦ Subtype.ext_iff


/-- The kernel of a homomorphism to a domain is a prime ideal. -/
theorem ker_isPrime {F : Type*} [Ring R] [Ring S] [IsDomain S]
    [FunLike F R S] [RingHomClass F R S] (f : F) :
    (ker f).IsPrime :=
  ⟨by
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝⁴ : Ring R
      inst✝³ : Ring S
      inst✝² : IsDomain S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      f : F
      ⊢ Ne (RingHom.ker f) Top.top
    -/
    rw [Ne, Ideal.eq_top_iff_one]
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝⁴ : Ring R
      inst✝³ : Ring S
      inst✝² : IsDomain S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      f : F
      ⊢ Not (Membership.mem (RingHom.ker f) 1)
    -/
    exact not_one_mem_ker f,
    /-
      🎉 no goals
    -/
   fun {x y} => by
    /-
      R : Type u
      S : Type v
      F : Type u_1
      inst✝⁴ : Ring R
      inst✝³ : Ring S
      inst✝² : IsDomain S
      inst✝¹ : FunLike F R S
      inst✝ : RingHomClass F R S
      f : F
      x y : R
      ⊢ Membership.mem (RingHom.ker f) (HMul.hMul x y) → Or (Membership.mem (RingHom …
    -/
    simpa only [mem_ker, map_mul] using @eq_zero_or_eq_zero_of_mul_eq_zero S _ _ _ _ _⟩
    /-
      🎉 no goals
    -/


/-- The kernel of a homomorphism to a field is a maximal ideal. -/
theorem ker_isMaximal_of_surjective {R K F : Type*} [Ring R] [Field K]
    [FunLike F R K] [RingHomClass F R K] (f : F)
    (hf : Function.Surjective f) : (ker f).IsMaximal := by
  refine
    Ideal.isMaximal_iff.mpr
      ⟨fun h1 => one_ne_zero' K <| map_one f ▸ mem_ker.mp h1, fun J x hJ hxf hxJ => ?_⟩
  /-
    R : Type u_1
    K : Type u_2
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Field K
    inst✝¹ : FunLike F R K
    inst✝ : RingHomClass F R K
    f : F
    hf : Function.Surjective ⇑f
    J : Ideal R
    x : R
    hJ : LE.le (RingHom.ker f) J
    hxf : Not (Membership.mem (RingHom.ker f) x)
    hxJ : Membership.mem J x
    ⊢ Membership.mem J 1
  -/
  obtain ⟨y, hy⟩ := hf (f x)⁻¹
  /-
    case intro
    R : Type u_1
    K : Type u_2
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Field K
    inst✝¹ : FunLike F R K
    inst✝ : RingHomClass F R K
    f : F
    hf : Function.Surjective ⇑f
    J : Ideal R
    x : R
    hJ : LE.le (RingHom.ker f) J
    hxf : Not (Membership.mem (RingHom.ker f) x)
    hxJ : Membership.mem J x
    y : R
    hy : Eq (f y) (Inv.inv (f x))
    ⊢ Membership.mem J 1
  -/
  have H : 1 = y * x - (y * x - 1) := (sub_sub_cancel _ _).symm
  /-
    case intro
    R : Type u_1
    K : Type u_2
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Field K
    inst✝¹ : FunLike F R K
    inst✝ : RingHomClass F R K
    f : F
    hf : Function.Surjective ⇑f
    J : Ideal R
    x : R
    hJ : LE.le (RingHom.ker f) J
    hxf : Not (Membership.mem (RingHom.ker f) x)
    hxJ : Membership.mem J x
    y : R
    hy : Eq (f y) (Inv.inv (f x))
    H : Eq 1 (HSub.hSub (HMul.hMul y x) (HSub.hSub (HMul.hMul y x) 1))
    ⊢ Membership.mem J 1
  -/
  rw [H]
  /-
    case intro
    R : Type u_1
    K : Type u_2
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Field K
    inst✝¹ : FunLike F R K
    inst✝ : RingHomClass F R K
    f : F
    hf : Function.Surjective ⇑f
    J : Ideal R
    x : R
    hJ : LE.le (RingHom.ker f) J
    hxf : Not (Membership.mem (RingHom.ker f) x)
    hxJ : Membership.mem J x
    y : R
    hy : Eq (f y) (Inv.inv (f x))
    H : Eq 1 (HSub.hSub (HMul.hMul y x) (HSub.hSub (HMul.hMul y x) 1))
    ⊢ Membership.mem J (HSub.hSub (HMul.hMul y x) (HSub.hSub (HMul.hMul y x) 1))
  -/
  refine J.sub_mem (J.mul_mem_left _ hxJ) (hJ ?_)
  /-
    case intro
    R : Type u_1
    K : Type u_2
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Field K
    inst✝¹ : FunLike F R K
    inst✝ : RingHomClass F R K
    f : F
    hf : Function.Surjective ⇑f
    J : Ideal R
    x : R
    hJ : LE.le (RingHom.ker f) J
    hxf : Not (Membership.mem (RingHom.ker f) x)
    hxJ : Membership.mem J x
    y : R
    hy : Eq (f y) (Inv.inv (f x))
    H : Eq 1 (HSub.hSub (HMul.hMul y x) (HSub.hSub (HMul.hMul y x) 1))
    ⊢ Membership.mem (RingHom.ker f) (HSub.hSub (HMul.hMul y x) 1)
  -/
  rw [mem_ker]
  /-
    case intro
    R : Type u_1
    K : Type u_2
    F : Type u_3
    inst✝³ : Ring R
    inst✝² : Field K
    inst✝¹ : FunLike F R K
    inst✝ : RingHomClass F R K
    f : F
    hf : Function.Surjective ⇑f
    J : Ideal R
    x : R
    hJ : LE.le (RingHom.ker f) J
    hxf : Not (Membership.mem (RingHom.ker f) x)
    hxJ : Membership.mem J x
    y : R
    hy : Eq (f y) (Inv.inv (f x))
    H : Eq 1 (HSub.hSub (HMul.hMul y x) (HSub.hSub (HMul.hMul y x) 1))
    ⊢ Eq (f (HSub.hSub (HMul.hMul y x) 1)) 0
  -/
  simp only [hy, map_sub, map_one, map_mul, inv_mul_cancel₀ (mt mem_ker.mpr hxf :), sub_self]
  /-
    🎉 no goals
  -/


variable (R M) in
/-- `Module.annihilator R M` is the ideal of all elements `r : R` such that `r • M = 0`. -/
def Module.annihilator : Ideal R := RingHom.ker (Module.toAddMonoidEnd R M)


theorem Module.mem_annihilator {r} : r ∈ Module.annihilator R M ↔ ∀ m : M, r • m = 0 :=
  ⟨fun h ↦ (congr($h ·)), (AddMonoidHom.ext ·)⟩


theorem LinearMap.annihilator_le_of_injective (f : M →ₗ[R] M') (hf : Function.Injective f) :
    Module.annihilator R M' ≤ Module.annihilator R M := fun x h ↦ by
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Injective ⇑f
    x : R
    h : Membership.mem (Module.annihilator R M') x
    ⊢ Membership.mem (Module.annihilator R M) x
  -/
  rw [Module.mem_annihilator] at h ⊢; exact fun m ↦ hf (by rw [map_smul, h, f.map_zero])
                                      /-
                                        🎉 no goals
                                      -/


theorem LinearMap.annihilator_le_of_surjective (f : M →ₗ[R] M')
    (hf : Function.Surjective f) : Module.annihilator R M ≤ Module.annihilator R M' := fun x h ↦ by
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Surjective ⇑f
    x : R
    h : Membership.mem (Module.annihilator R M) x
    ⊢ Membership.mem (Module.annihilator R M') x
  -/
  rw [Module.mem_annihilator] at h ⊢
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Surjective ⇑f
    x : R
    h : ∀ (m : M), Eq (HSMul.hSMul x m) 0
    ⊢ ∀ (m : M'), Eq (HSMul.hSMul x m) 0
  -/
  intro m; obtain ⟨m, rfl⟩ := hf m
  /-
    case intro
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Surjective ⇑f
    x : R
    h : ∀ (m : M), Eq (HSMul.hSMul x m) 0
    m : M
    ⊢ Eq (HSMul.hSMul x (f m)) 0
  -/
  rw [← map_smul, h, f.map_zero]
  /-
    🎉 no goals
  -/


theorem LinearEquiv.annihilator_eq (e : M ≃ₗ[R] M') :
    Module.annihilator R M = Module.annihilator R M' :=
  (e.annihilator_le_of_surjective e.surjective).antisymm (e.annihilator_le_of_injective e.injective)


theorem Module.comap_annihilator {R₀} [CommSemiring R₀] [Module R₀ M]
    [Algebra R₀ R] [IsScalarTower R₀ R M] :
    (Module.annihilator R M).comap (algebraMap R₀ R) = Module.annihilator R₀ M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    R₀ : Type u_4
    inst✝³ : CommSemiring R₀
    inst✝² : Module R₀ M
    inst✝¹ : Algebra R₀ R
    inst✝ : IsScalarTower R₀ R M
    ⊢ Eq (Ideal.comap (algebraMap R₀ R) (Module.annihilator R M)) (Module.annihila …
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    R₀ : Type u_4
    inst✝³ : CommSemiring R₀
    inst✝² : Module R₀ M
    inst✝¹ : Algebra R₀ R
    inst✝ : IsScalarTower R₀ R M
    x : R₀
    ⊢ Iff (Membership.mem (Ideal.comap (algebraMap R₀ R) (Module.annihilator R M)) …
  -/
  simp [mem_annihilator]
  /-
    🎉 no goals
  -/


lemma Module.annihilator_eq_bot {R M} [Ring R] [AddCommGroup M] [Module R M] :
    Module.annihilator R M = ⊥ ↔ FaithfulSMul R M := by
  /-
    R : Type u_4
    M : Type u_5
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (Eq (Module.annihilator R M) Bot.bot) (FaithfulSMul R M)
  -/
  rw [← le_bot_iff]
  /-
    R : Type u_4
    M : Type u_5
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (LE.le (Module.annihilator R M) Bot.bot) (FaithfulSMul R M)
  -/
  refine ⟨fun H ↦ ⟨fun {r s} H' ↦ ?_⟩, fun ⟨H⟩ {a} ha ↦ ?_⟩
    /-
      case refine_1
      R : Type u_4
      M : Type u_5
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      H : LE.le (Module.annihilator R M) Bot.bot
      r s : R
      H' : ∀ (a : M), Eq (HSMul.hSMul r a) (HSMul.hSMul s a)
      ⊢ Eq r s
    -/
  · rw [← sub_eq_zero]
    exact H (Module.mem_annihilator (r := r - s).mpr
      (by simp only [sub_smul, H', sub_self, implies_true]))
    /-
      case refine_2
      R : Type u_4
      M : Type u_5
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ : FaithfulSMul R M
      a : R
      ha : Membership.mem (Module.annihilator R M) a
      H : ∀ {m₁ m₂ : R}, (∀ (a : M), Eq (HSMul.hSMul m₁ a) (HSMul.hSMul m₂ a)) → Eq  …
      ⊢ Membership.mem Bot.bot a
    -/
  · exact @H a 0 (by simp [Module.mem_annihilator.mp ha])
    /-
      🎉 no goals
    -/


/-- `N.annihilator` is the ideal of all elements `r : R` such that `r • N = 0`. -/
abbrev annihilator (N : Submodule R M) : Ideal R :=
  Module.annihilator R N


theorem annihilator_top : (⊤ : Submodule R M).annihilator = Module.annihilator R M :=
  topEquiv.annihilator_eq


theorem mem_annihilator {r} : r ∈ N.annihilator ↔ ∀ n ∈ N, r • n = (0 : M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    r : R
    ⊢ Iff (Membership.mem N.annihilator r) (∀ (n : M), Membership.mem N n → Eq (HS …
  -/
  simp_rw [annihilator, Module.mem_annihilator, Subtype.forall, Subtype.ext_iff]; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem annihilator_bot : (⊥ : Submodule R M).annihilator = ⊤ :=
                                                         /-
                                                           R : Type u_1
                                                           M : Type u_2
                                                           inst✝² : Semiring R
                                                           inst✝¹ : AddCommMonoid M
                                                           inst✝ : Module R M
                                                           x✝² : R
                                                           x✝¹ : Membership.mem Top.top x✝²
                                                           x✝ : M
                                                           ⊢ Membership.mem Bot.bot x✝ → Eq (HSMul.hSMul x✝² x✝) 0
                                                         -/
  top_le_iff.mp fun _ _ ↦ mem_annihilator.mpr fun _ ↦ by rintro rfl; rw [smul_zero]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem annihilator_eq_top_iff : N.annihilator = ⊤ ↔ N = ⊥ :=
  ⟨fun H ↦
    eq_bot_iff.2 fun (n : M) hn =>
      (mem_bot R).2 <| one_smul R n ▸ mem_annihilator.1 ((Ideal.eq_top_iff_one _).1 H) n hn,
    fun H ↦ H.symm ▸ annihilator_bot⟩


theorem annihilator_mono (h : N ≤ P) : P.annihilator ≤ N.annihilator := fun _ hrp =>
  mem_annihilator.2 fun n hn => mem_annihilator.1 hrp n <| h hn


theorem annihilator_iSup (ι : Sort w) (f : ι → Submodule R M) :
    annihilator (⨆ i, f i) = ⨅ i, annihilator (f i) :=
  le_antisymm (le_iInf fun _ => annihilator_mono <| le_iSup _ _) fun r H =>
    mem_annihilator.2 fun n hn ↦ iSup_induction f (C := (r • · = 0)) hn
      (fun i ↦ mem_annihilator.1 <| (mem_iInf _).mp H i) (smul_zero _)
                           /-
                             R : Type u_1
                             M : Type u_2
                             inst✝² : Semiring R
                             inst✝¹ : AddCommMonoid M
                             inst✝ : Module R M
                             ι : Sort w
                             f : ι → Submodule R M
                             r : R
                             H : Membership.mem (iInf fun i => (f i).annihilator) r
                             n : M
                             hn : Membership.mem (iSup fun i => f i) n
                             m₁ m₂ : M
                             h₁ : (fun x => Eq (HSMul.hSMul r x) 0) m₁
                             h₂ : (fun x => Eq (HSMul.hSMul r x) 0) m₂
                             ⊢ (fun x => Eq (HSMul.hSMul r x) 0) (HAdd.hAdd m₁ m₂)
                           -/
      fun m₁ m₂ h₁ h₂ ↦ by simp_rw [smul_add, h₁, h₂, add_zero]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem annihilator_smul (N : Submodule R M) : annihilator N • N = ⊥ :=
  eq_bot_iff.2 (smul_le.2 fun _ => mem_annihilator.1)


@[simp]
theorem annihilator_mul (I : Ideal R) : annihilator I * I = ⊥ :=
  annihilator_smul I


theorem mem_annihilator' {r} : r ∈ N.annihilator ↔ N ≤ comap (r • (LinearMap.id : M →ₗ[R] M)) ⊥ :=
  mem_annihilator.trans ⟨fun H n hn => (mem_bot R).2 <| H n hn, fun H _ hn => (mem_bot R).1 <| H hn⟩


theorem mem_annihilator_span (s : Set M) (r : R) :
    r ∈ (Submodule.span R s).annihilator ↔ ∀ n : s, r • (n : M) = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    r : R
    ⊢ Iff (Membership.mem (Submodule.span R s).annihilator r) (∀ (n : ↑s), Eq (HSM …
  -/
  rw [Submodule.mem_annihilator]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    r : R
    ⊢ Iff (∀ (n : M), Membership.mem (Submodule.span R s) n → Eq (HSMul.hSMul r n) …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      ⊢ (∀ (n : M), Membership.mem (Submodule.span R s) n → Eq (HSMul.hSMul r n) 0)  …
    -/
  · intro h n
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      h : ∀ (n : M), Membership.mem (Submodule.span R s) n → Eq (HSMul.hSMul r n) 0
      n : ↑s
      ⊢ Eq (HSMul.hSMul r ↑n) 0
    -/
    exact h _ (Submodule.subset_span n.prop)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      ⊢ (∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0) → ∀ (n : M), Membership.mem (Submodule …
    -/
  · intro h n hn
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
      n : M
      hn : Membership.mem (Submodule.span R s) n
      ⊢ Eq (HSMul.hSMul r n) 0
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hn
      /-
        case mpr.refine_1
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        ⊢ ∀ (x : M), Membership.mem s x → Eq (HSMul.hSMul r x) 0
      -/
    · intro x hx
      /-
        case mpr.refine_1
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        x : M
        hx : Membership.mem s x
        ⊢ Eq (HSMul.hSMul r x) 0
      -/
      exact h ⟨x, hx⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        ⊢ Eq (HSMul.hSMul r 0) 0
      -/
    · exact smul_zero _
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_3
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        ⊢ ∀ (x y : M), Membership.mem (Submodule.span R s) x → Membership.mem (Submodu …
      -/
    · intro x y _ _ hx hy
      /-
        case mpr.refine_3
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        x y : M
        hx✝ : Membership.mem (Submodule.span R s) x
        hy✝ : Membership.mem (Submodule.span R s) y
        hx : Eq (HSMul.hSMul r x) 0
        hy : Eq (HSMul.hSMul r y) 0
        ⊢ Eq (HSMul.hSMul r (HAdd.hAdd x y)) 0
      -/
      rw [smul_add, hx, hy, zero_add]
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_4
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        ⊢ ∀ (a : R) (x : M), Membership.mem (Submodule.span R s) x → Eq (HSMul.hSMul r …
      -/
    · intro a x _ hx
      /-
        case mpr.refine_4
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        s : Set M
        r : R
        h : ∀ (n : ↑s), Eq (HSMul.hSMul r ↑n) 0
        n : M
        hn : Membership.mem (Submodule.span R s) n
        a : R
        x : M
        hx✝ : Membership.mem (Submodule.span R s) x
        hx : Eq (HSMul.hSMul r x) 0
        ⊢ Eq (HSMul.hSMul r (HSMul.hSMul a x)) 0
      -/
      rw [smul_comm, hx, smul_zero]
      /-
        🎉 no goals
      -/


theorem mem_annihilator_span_singleton (g : M) (r : R) :
                                                                       /-
                                                                         R : Type u_1
                                                                         M : Type u_2
                                                                         inst✝² : CommSemiring R
                                                                         inst✝¹ : AddCommMonoid M
                                                                         inst✝ : Module R M
                                                                         g : M
                                                                         r : R
                                                                         ⊢ Iff (Membership.mem (Submodule.span R (Singleton.singleton g)).annihilator r …
                                                                       -/
    r ∈ (Submodule.span R ({g} : Set M)).annihilator ↔ r • g = 0 := by simp [mem_annihilator_span]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝ : CommSemiring R
                                                                      I : Ideal R
                                                                      ⊢ Eq (HMul.hMul I (Submodule.annihilator I)) Bot.bot
                                                                    -/
theorem mul_annihilator (I : Ideal R) : I * annihilator I = ⊥ := by rw [mul_comm, annihilator_mul]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem map_eq_bot_iff_le_ker {I : Ideal R} (f : F) : I.map f = ⊥ ↔ I ≤ RingHom.ker f := by
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    I : Ideal R
    f : F
    ⊢ Iff (Eq (Ideal.map f I) Bot.bot) (LE.le I (RingHom.ker f))
  -/
  rw [RingHom.ker, eq_bot_iff, map_le_iff_le_comap]
  /-
    🎉 no goals
  -/


theorem ker_le_comap {K : Ideal S} (f : F) : RingHom.ker f ≤ comap f K := fun _ hx =>
  mem_comap.2 (RingHom.mem_ker.1 hx ▸ K.zero_mem)


/-- A ring isomorphism sends a prime ideal to a prime ideal. -/
instance map_isPrime_of_equiv {F' : Type*} [EquivLike F' R S] [RingEquivClass F' R S]
    (f : F') {I : Ideal R} [IsPrime I] : IsPrime (map f I) := by
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring S
    inst✝³ : FunLike F R S
    rc : RingHomClass F R S
    F' : Type u_4
    inst✝² : EquivLike F' R S
    inst✝¹ : RingEquivClass F' R S
    f : F'
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ (Ideal.map f I).IsPrime
  -/
  have h : I.map f = I.map ((f : R ≃+* S) : R →+* S) := rfl
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring S
    inst✝³ : FunLike F R S
    rc : RingHomClass F R S
    F' : Type u_4
    inst✝² : EquivLike F' R S
    inst✝¹ : RingEquivClass F' R S
    f : F'
    I : Ideal R
    inst✝ : I.IsPrime
    h : Eq (Ideal.map f I) (Ideal.map (↑↑f) I)
    ⊢ (Ideal.map f I).IsPrime
  -/
  rw [h, map_comap_of_equiv (f : R ≃+* S)]
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring S
    inst✝³ : FunLike F R S
    rc : RingHomClass F R S
    F' : Type u_4
    inst✝² : EquivLike F' R S
    inst✝¹ : RingEquivClass F' R S
    f : F'
    I : Ideal R
    inst✝ : I.IsPrime
    h : Eq (Ideal.map f I) (Ideal.map (↑↑f) I)
    ⊢ (Ideal.comap (↑f).symm I).IsPrime
  -/
  exact Ideal.IsPrime.comap (RingEquiv.symm (f : R ≃+* S))
  /-
    🎉 no goals
  -/


lemma comap_map_of_surjective' (f : F) (hf : Function.Surjective f) (I : Ideal R) :
    (I.map f).comap f = I ⊔ RingHom.ker f :=
  comap_map_of_surjective f hf I


theorem map_sInf {A : Set (Ideal R)} {f : F} (hf : Function.Surjective f) :
    (∀ J ∈ A, RingHom.ker f ≤ J) → map f (sInf A) = sInf (map f '' A) := by
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    A : Set (Ideal R)
    f : F
    hf : Function.Surjective ⇑f
    ⊢ (∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J) → Eq (Ideal. …
  -/
  refine fun h => le_antisymm (le_sInf ?_) ?_
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      ⊢ ∀ (b : Ideal S), Membership.mem (Set.image (Ideal.map f) A) b → LE.le (Ideal …
    -/
  · intro j hj y hy
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      j : Ideal S
      hj : Membership.mem (Set.image (Ideal.map f) A) j
      y : S
      hy : Membership.mem (Ideal.map f (InfSet.sInf A)) y
      ⊢ Membership.mem j y
    -/
    cases' (mem_map_iff_of_surjective f hf).1 hy with x hx
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      j : Ideal S
      hj : Membership.mem (Set.image (Ideal.map f) A) j
      y : S
      hy : Membership.mem (Ideal.map f (InfSet.sInf A)) y
      x : R
      hx : And (Membership.mem (InfSet.sInf A) x) (Eq (f x) y)
      ⊢ Membership.mem j y
    -/
    cases' (Set.mem_image _ _ _).mp hj with J hJ
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      j : Ideal S
      hj : Membership.mem (Set.image (Ideal.map f) A) j
      y : S
      hy : Membership.mem (Ideal.map f (InfSet.sInf A)) y
      x : R
      hx : And (Membership.mem (InfSet.sInf A) x) (Eq (f x) y)
      J : Ideal R
      hJ : And (Membership.mem A J) (Eq (Ideal.map f J) j)
      ⊢ Membership.mem j y
    -/
    rw [← hJ.right, ← hx.right]
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      j : Ideal S
      hj : Membership.mem (Set.image (Ideal.map f) A) j
      y : S
      hy : Membership.mem (Ideal.map f (InfSet.sInf A)) y
      x : R
      hx : And (Membership.mem (InfSet.sInf A) x) (Eq (f x) y)
      J : Ideal R
      hJ : And (Membership.mem A J) (Eq (Ideal.map f J) j)
      ⊢ Membership.mem (Ideal.map f J) (f x)
    -/
    exact mem_map_of_mem f (sInf_le_of_le hJ.left (le_of_eq rfl) hx.left)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      ⊢ LE.le (InfSet.sInf (Set.image (Ideal.map f) A)) (Ideal.map f (InfSet.sInf A))
    -/
  · intro y hy
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      y : S
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) y
      ⊢ Membership.mem (Ideal.map f (InfSet.sInf A)) y
    -/
    cases' hf y with x hx
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      y : S
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) y
      x : R
      hx : Eq (f x) y
      ⊢ Membership.mem (Ideal.map f (InfSet.sInf A)) y
    -/
    refine hx ▸ mem_map_of_mem f ?_
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      y : S
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) y
      x : R
      hx : Eq (f x) y
      ⊢ Membership.mem (InfSet.sInf A) x
    -/
    have : ∀ I ∈ A, y ∈ map f I := by simpa using hy
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      y : S
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) y
      x : R
      hx : Eq (f x) y
      this : ∀ (I : Ideal R), Membership.mem A I → Membership.mem (Ideal.map f I) y
      ⊢ Membership.mem (InfSet.sInf A) x
    -/
    rw [Submodule.mem_sInf]
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      y : S
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) y
      x : R
      hx : Eq (f x) y
      this : ∀ (I : Ideal R), Membership.mem A I → Membership.mem (Ideal.map f I) y
      ⊢ ∀ (p : Submodule R R), Membership.mem A p → Membership.mem p x
    -/
    intro J hJ
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      y : S
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) y
      x : R
      hx : Eq (f x) y
      this : ∀ (I : Ideal R), Membership.mem A I → Membership.mem (Ideal.map f I) y
      J : Submodule R R
      hJ : Membership.mem A J
      ⊢ Membership.mem J x
    -/
    rcases (mem_map_iff_of_surjective f hf).1 (this J hJ) with ⟨x', hx', rfl⟩
    have : x - x' ∈ J := by
      apply h J hJ
      rw [RingHom.mem_ker, map_sub, hx, sub_self]
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      A : Set (Ideal R)
      f : F
      hf : Function.Surjective ⇑f
      h : ∀ (J : Ideal R), Membership.mem A J → LE.le (RingHom.ker f) J
      x : R
      J : Submodule R R
      hJ : Membership.mem A J
      x' : R
      hx' : Membership.mem J x'
      hy : Membership.mem (InfSet.sInf (Set.image (Ideal.map f) A)) (f x')
      hx : Eq (f x) (f x')
      this✝ : ∀ (I : Ideal R), Membership.mem A I → Membership.mem (Ideal.map f I) ( …
      this : Membership.mem J (HSub.hSub x x')
      ⊢ Membership.mem J x
    -/
    simpa only [sub_add_cancel] using J.add_mem this hx'
    /-
      🎉 no goals
    -/


theorem map_isPrime_of_surjective {f : F} (hf : Function.Surjective f) {I : Ideal R} [H : IsPrime I]
    (hk : RingHom.ker f ≤ I) : IsPrime (map f I) := by
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    I : Ideal R
    H : I.IsPrime
    hk : LE.le (RingHom.ker f) I
    ⊢ (Ideal.map f I).IsPrime
  -/
  refine ⟨fun h => H.ne_top (eq_top_iff.2 ?_), fun {x y} => ?_⟩
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      h : Eq (Ideal.map f I) Top.top
      ⊢ LE.le Top.top I
    -/
  · replace h := congr_arg (comap f) h
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      h : Eq (Ideal.comap f (Ideal.map f I)) (Ideal.comap f Top.top)
      ⊢ LE.le Top.top I
    -/
    rw [comap_map_of_surjective _ hf, comap_top] at h
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      h : Eq (Max.max I (Ideal.comap f Bot.bot)) Top.top
      ⊢ LE.le Top.top I
    -/
    exact h ▸ sup_le (le_of_eq rfl) hk
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      x y : S
      ⊢ Membership.mem (Ideal.map f I) (HMul.hMul x y) → Or (Membership.mem (Ideal.m …
    -/
  · refine fun hxy => (hf x).recOn fun a ha => (hf y).recOn fun b hb => ?_
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      x y : S
      hxy : Membership.mem (Ideal.map f I) (HMul.hMul x y)
      a : R
      ha : Eq (f a) x
      b : R
      hb : Eq (f b) y
      ⊢ Or (Membership.mem (Ideal.map f I) x) (Membership.mem (Ideal.map f I) y)
    -/
    rw [← ha, ← hb, ← _root_.map_mul f, mem_map_iff_of_surjective _ hf] at hxy
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      x y : S
      a : R
      ha : Eq (f a) x
      b : R
      hxy : Exists fun x => And (Membership.mem I x) (Eq (f x) (f (HMul.hMul a b)))
      hb : Eq (f b) y
      ⊢ Or (Membership.mem (Ideal.map f I) x) (Membership.mem (Ideal.map f I) y)
    -/
    rcases hxy with ⟨c, hc, hc'⟩
    /-
      case refine_2.intro.intro
      R : Type u_1
      S : Type u_2
      F : Type u_3
      inst✝² : Ring R
      inst✝¹ : Ring S
      inst✝ : FunLike F R S
      rc : RingHomClass F R S
      f : F
      hf : Function.Surjective ⇑f
      I : Ideal R
      H : I.IsPrime
      hk : LE.le (RingHom.ker f) I
      x y : S
      a : R
      ha : Eq (f a) x
      b : R
      hb : Eq (f b) y
      c : R
      hc : Membership.mem I c
      hc' : Eq (f c) (f (HMul.hMul a b))
      ⊢ Or (Membership.mem (Ideal.map f I) x) (Membership.mem (Ideal.map f I) y)
    -/
    rw [← sub_eq_zero, ← map_sub] at hc'
    have : a * b ∈ I := by
      convert I.sub_mem hc (hk (hc' : c - a * b ∈ RingHom.ker f)) using 1
      abel
    exact
      (H.mem_or_mem this).imp (fun h => ha ▸ mem_map_of_mem f h) fun h => hb ▸ mem_map_of_mem f h


theorem map_eq_bot_iff_of_injective {I : Ideal R} {f : F} (hf : Function.Injective f) :
    I.map f = ⊥ ↔ I = ⊥ := by
  /-
    R : Type u_1
    S : Type u_2
    F : Type u_3
    inst✝² : Ring R
    inst✝¹ : Ring S
    inst✝ : FunLike F R S
    rc : RingHomClass F R S
    I : Ideal R
    f : F
    hf : Function.Injective ⇑f
    ⊢ Iff (Eq (Ideal.map f I) Bot.bot) (Eq I Bot.bot)
  -/
  rw [map_eq_bot_iff_le_ker, (RingHom.injective_iff_ker_eq_bot f).mp hf, le_bot_iff]
  /-
    🎉 no goals
  -/


theorem map_ne_bot_of_ne_bot {S : Type*} [Ring S] [Nontrivial S] [Algebra R S]
    [NoZeroSMulDivisors R S] {I : Ideal R} (h : I ≠ ⊥) : map (algebraMap R S) I ≠ ⊥ :=
  (map_eq_bot_iff_of_injective (NoZeroSMulDivisors.algebraMap_injective R S)).mp.mt h


theorem map_eq_iff_sup_ker_eq_of_surjective {I J : Ideal R} (f : R →+* S)
    (hf : Function.Surjective f) : map f I = map f J ↔ I ⊔ RingHom.ker f = J ⊔ RingHom.ker f := by
  rw [← (comap_injective_of_surjective f hf).eq_iff, comap_map_of_surjective f hf,
    comap_map_of_surjective f hf, RingHom.ker_eq_comap_bot]


theorem map_radical_of_surjective {f : R →+* S} (hf : Function.Surjective f) {I : Ideal R}
    (h : RingHom.ker f ≤ I) : map f I.radical = (map f I).radical := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal R
    h : LE.le (RingHom.ker f) I
    ⊢ Eq (Ideal.map f I.radical) (Ideal.map f I).radical
  -/
  rw [radical_eq_sInf, radical_eq_sInf]
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal R
    h : LE.le (RingHom.ker f) I
    ⊢ Eq (Ideal.map f (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime))) (I …
  -/
  have : ∀ J ∈ {J : Ideal R | I ≤ J ∧ J.IsPrime}, RingHom.ker f ≤ J := fun J hJ => h.trans hJ.left
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal R
    h : LE.le (RingHom.ker f) I
    this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPri …
    ⊢ Eq (Ideal.map f (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsPrime))) (I …
  -/
  convert map_sInf hf this
  /-
    case h.e'_3.h.e'_3
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    I : Ideal R
    h : LE.le (RingHom.ker f) I
    this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPri …
    ⊢ Eq (setOf fun J => And (LE.le (Ideal.map f I) J) J.IsPrime) (Set.image (Idea …
  -/
  refine funext fun j => propext ⟨?_, ?_⟩
    /-
      case h.e'_3.h.e'_3.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal R
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPri …
      j : Ideal S
      ⊢ setOf (fun J => And (LE.le (Ideal.map f I) J) J.IsPrime) j → Set.image (Idea …
    -/
  · rintro ⟨hj, hj'⟩
    /-
      case h.e'_3.h.e'_3.refine_1.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal R
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPri …
      j : Ideal S
      hj : LE.le (Ideal.map f I) j
      hj' : j.IsPrime
      ⊢ Set.image (Ideal.map f) (setOf fun J => And (LE.le I J) J.IsPrime) j
    -/
    haveI : j.IsPrime := hj'
    exact
      ⟨comap f j, ⟨⟨map_le_iff_le_comap.1 hj, comap_isPrime f j⟩, map_comap_of_surjective f hf j⟩⟩
    /-
      case h.e'_3.h.e'_3.refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal R
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPri …
      j : Ideal S
      ⊢ Set.image (Ideal.map f) (setOf fun J => And (LE.le I J) J.IsPrime) j → setOf …
    -/
  · rintro ⟨J, ⟨hJ, hJ'⟩⟩
    /-
      case h.e'_3.h.e'_3.refine_2.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal R
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPri …
      j : Ideal S
      J : Ideal R
      hJ : Membership.mem (setOf fun J => And (LE.le I J) J.IsPrime) J
      hJ' : Eq (Ideal.map f J) j
      ⊢ setOf (fun J => And (LE.le (Ideal.map f I) J) J.IsPrime) j
    -/
    haveI : J.IsPrime := hJ.right
    /-
      case h.e'_3.h.e'_3.refine_2.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      I : Ideal R
      h : LE.le (RingHom.ker f) I
      this✝ : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsPr …
      j : Ideal S
      J : Ideal R
      hJ : Membership.mem (setOf fun J => And (LE.le I J) J.IsPrime) J
      hJ' : Eq (Ideal.map f J) j
      this : J.IsPrime
      ⊢ setOf (fun J => And (LE.le (Ideal.map f I) J) J.IsPrime) j
    -/
    exact ⟨hJ' ▸ map_mono hJ.left, hJ' ▸ map_isPrime_of_surjective hf (le_trans h hJ.left)⟩
    /-
      🎉 no goals
    -/


/-- Auxiliary definition used to define `liftOfRightInverse` -/
def liftOfRightInverseAux (hf : Function.RightInverse f_inv f) (g : A →+* C)
    (hg : RingHom.ker f ≤ RingHom.ker g) :
    B →+* C :=
  { AddMonoidHom.liftOfRightInverse f.toAddMonoidHom f_inv hf ⟨g.toAddMonoidHom, hg⟩ with
    toFun := fun b => g (f_inv b)
    map_one' := by
      /-
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        ⊢ Eq ((fun b => g (f_inv b)) 1) 1
      -/
      rw [← map_one g, ← sub_eq_zero, ← map_sub g, ← mem_ker]
      /-
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        ⊢ Membership.mem (RingHom.ker g) (HSub.hSub (f_inv 1) 1)
      -/
      apply hg
      /-
        case a
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        ⊢ Membership.mem (RingHom.ker f) (HSub.hSub (f_inv 1) 1)
      -/
      rw [mem_ker, map_sub f, sub_eq_zero, map_one f]
      /-
        case a
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        ⊢ Eq (f (f_inv 1)) 1
      -/
      exact hf 1
      /-
        🎉 no goals
      -/
    map_mul' := by
      /-
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        ⊢ ∀ (x y : B), Eq ({ toFun := fun b => g (f_inv b), map_one' := ⋯ }.toFun (HMu …
      -/
      intro x y
      /-
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        x y : B
        ⊢ Eq ({ toFun := fun b => g (f_inv b), map_one' := ⋯ }.toFun (HMul.hMul x y))  …
      -/
      rw [← map_mul g, ← sub_eq_zero, ← map_sub g, ← mem_ker]
      /-
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        x y : B
        ⊢ Membership.mem (RingHom.ker g) (HSub.hSub (f_inv (HMul.hMul x y)) (HMul.hMul …
      -/
      apply hg
      /-
        case a
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        x y : B
        ⊢ Membership.mem (RingHom.ker f) (HSub.hSub (f_inv (HMul.hMul x y)) (HMul.hMul …
      -/
      rw [mem_ker, map_sub f, sub_eq_zero, map_mul f]
      /-
        case a
        A : Type u_1
        B : Type u_2
        C : Type u_3
        inst✝² : Ring A
        inst✝¹ : Ring B
        inst✝ : Ring C
        f : RingHom A B
        f_inv : B → A
        hf : Function.RightInverse f_inv ⇑f
        g : RingHom A C
        hg : LE.le (RingHom.ker f) (RingHom.ker g)
        x y : B
        ⊢ Eq (f (f_inv (HMul.hMul x y))) (HMul.hMul (f (f_inv x)) (f (f_inv y)))
      -/
      simp only [hf _] }
      /-
        🎉 no goals
      -/


@[simp]
theorem liftOfRightInverseAux_comp_apply (hf : Function.RightInverse f_inv f) (g : A →+* C)
    (hg : RingHom.ker f ≤ RingHom.ker g) (a : A) :
    (f.liftOfRightInverseAux f_inv hf g hg) (f a) = g a :=
  f.toAddMonoidHom.liftOfRightInverse_comp_apply f_inv hf ⟨g.toAddMonoidHom, hg⟩ a


/-- `liftOfRightInverse f hf g hg` is the unique ring homomorphism `φ`

* such that `φ.comp f = g` (`RingHom.liftOfRightInverse_comp`),
* where `f : A →+* B` has a right_inverse `f_inv` (`hf`),
* and `g : B →+* C` satisfies `hg : f.ker ≤ g.ker`.

See `RingHom.eq_liftOfRightInverse` for the uniqueness lemma.

```
   A .
   |  \
 f |   \ g
   |    \
   v     \⌟
   B ----> C
      ∃!φ
```
-/
def liftOfRightInverse (hf : Function.RightInverse f_inv f) :
    { g : A →+* C // RingHom.ker f ≤ RingHom.ker g } ≃ (B →+* C) where
  toFun g := f.liftOfRightInverseAux f_inv hf g.1 g.2
                                                       /-
                                                         A : Type u_1
                                                         B : Type u_2
                                                         C : Type u_3
                                                         inst✝² : Ring A
                                                         inst✝¹ : Ring B
                                                         inst✝ : Ring C
                                                         f : RingHom A B
                                                         f_inv : B → A
                                                         hf : Function.RightInverse f_inv ⇑f
                                                         φ : RingHom B C
                                                         x : A
                                                         hx : Membership.mem (RingHom.ker f) x
                                                         ⊢ Eq ((φ.comp f) x) 0
                                                       -/
  invFun φ := ⟨φ.comp f, fun x hx => mem_ker.mpr <| by simp [mem_ker.mp hx]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/
  left_inv g := by
    /-
      A : Type u_1
      B : Type u_2
      C : Type u_3
      inst✝² : Ring A
      inst✝¹ : Ring B
      inst✝ : Ring C
      f : RingHom A B
      f_inv : B → A
      hf : Function.RightInverse f_inv ⇑f
      g : Subtype fun g => LE.le (RingHom.ker f) (RingHom.ker g)
      ⊢ Eq ((fun φ => ⟨φ.comp f, ⋯⟩) ((fun g => f.liftOfRightInverseAux f_inv hf ↑g  …
    -/
    ext
    /-
      case a.a
      A : Type u_1
      B : Type u_2
      C : Type u_3
      inst✝² : Ring A
      inst✝¹ : Ring B
      inst✝ : Ring C
      f : RingHom A B
      f_inv : B → A
      hf : Function.RightInverse f_inv ⇑f
      g : Subtype fun g => LE.le (RingHom.ker f) (RingHom.ker g)
      x✝ : A
      ⊢ Eq (↑((fun φ => ⟨φ.comp f, ⋯⟩) ((fun g => f.liftOfRightInverseAux f_inv hf ↑ …
    -/
    simp only [comp_apply, liftOfRightInverseAux_comp_apply, Subtype.coe_mk]
    /-
      🎉 no goals
    -/
  right_inv φ := by
    /-
      A : Type u_1
      B : Type u_2
      C : Type u_3
      inst✝² : Ring A
      inst✝¹ : Ring B
      inst✝ : Ring C
      f : RingHom A B
      f_inv : B → A
      hf : Function.RightInverse f_inv ⇑f
      φ : RingHom B C
      ⊢ Eq ((fun g => f.liftOfRightInverseAux f_inv hf ↑g ⋯) ((fun φ => ⟨φ.comp f, ⋯ …
    -/
    ext b
    /-
      case a
      A : Type u_1
      B : Type u_2
      C : Type u_3
      inst✝² : Ring A
      inst✝¹ : Ring B
      inst✝ : Ring C
      f : RingHom A B
      f_inv : B → A
      hf : Function.RightInverse f_inv ⇑f
      φ : RingHom B C
      b : B
      ⊢ Eq (((fun g => f.liftOfRightInverseAux f_inv hf ↑g ⋯) ((fun φ => ⟨φ.comp f,  …
    -/
    simp [liftOfRightInverseAux, hf b]
    /-
      🎉 no goals
    -/


/-- A non-computable version of `RingHom.liftOfRightInverse` for when no computable right
inverse is available, that uses `Function.surjInv`. -/
@[simp]
noncomputable abbrev liftOfSurjective (hf : Function.Surjective f) :
    { g : A →+* C // RingHom.ker f ≤ RingHom.ker g } ≃ (B →+* C) :=
  f.liftOfRightInverse (Function.surjInv hf) (Function.rightInverse_surjInv hf)


theorem liftOfRightInverse_comp_apply (hf : Function.RightInverse f_inv f)
    (g : { g : A →+* C // RingHom.ker f ≤ RingHom.ker g }) (x : A) :
    (f.liftOfRightInverse f_inv hf g) (f x) = g.1 x :=
  f.liftOfRightInverseAux_comp_apply f_inv hf g.1 g.2 x


theorem liftOfRightInverse_comp (hf : Function.RightInverse f_inv f)
    (g : { g : A →+* C // RingHom.ker f ≤ RingHom.ker g }) :
    (f.liftOfRightInverse f_inv hf g).comp f = g :=
  RingHom.ext <| f.liftOfRightInverse_comp_apply f_inv hf g


theorem eq_liftOfRightInverse (hf : Function.RightInverse f_inv f) (g : A →+* C)
    (hg : RingHom.ker f ≤ RingHom.ker g) (h : B →+* C) (hh : h.comp f = g) :
    h = f.liftOfRightInverse f_inv hf ⟨g, hg⟩ := by
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : Ring A
    inst✝¹ : Ring B
    inst✝ : Ring C
    f : RingHom A B
    f_inv : B → A
    hf : Function.RightInverse f_inv ⇑f
    g : RingHom A C
    hg : LE.le (RingHom.ker f) (RingHom.ker g)
    h : RingHom B C
    hh : Eq (h.comp f) g
    ⊢ Eq h ((f.liftOfRightInverse f_inv hf) ⟨g, hg⟩)
  -/
  simp_rw [← hh]
  /-
    A : Type u_1
    B : Type u_2
    C : Type u_3
    inst✝² : Ring A
    inst✝¹ : Ring B
    inst✝ : Ring C
    f : RingHom A B
    f_inv : B → A
    hf : Function.RightInverse f_inv ⇑f
    g : RingHom A C
    hg : LE.le (RingHom.ker f) (RingHom.ker g)
    h : RingHom B C
    hh : Eq (h.comp f) g
    ⊢ Eq h ((f.liftOfRightInverse f_inv hf) ⟨h.comp f, ⋯⟩)
  -/
  exact ((f.liftOfRightInverse f_inv hf).apply_symm_apply _).symm
  /-
    🎉 no goals
  -/


lemma coe_ker : RingHom.ker f = RingHom.ker (f : A →+* B) := rfl


lemma coe_ideal_map (I : Ideal A) :
    Ideal.map f I = Ideal.map (f : A →+* B) I := rfl


lemma comap_ker {C : Type*} [Semiring C] [Algebra R C] (f : B →ₐ[R] C) (g : A →ₐ[R] B) :
    (RingHom.ker f).comap g = RingHom.ker (f.comp g) :=
  RingHom.comap_ker f.toRingHom g.toRingHom


/-- The induced linear map from `I` to the span of `I` in an `R`-algebra `S`. -/
@[simps!]
def idealMap (I : Ideal R) : I →ₗ[R] I.map (algebraMap R S) :=
  (Algebra.linearMap R S).restrict (q := (I.map (algebraMap R S)).restrictScalars R)
    (fun _ ↦ Ideal.mem_map_of_mem _)


theorem of_ker_algebraMap_eq_bot (R A : Type*) [CommRing R] [Semiring A] [Algebra R A]
    [NoZeroDivisors A] (h : RingHom.ker (algebraMap R A) = ⊥) : NoZeroSMulDivisors R A :=
  of_algebraMap_injective ((RingHom.injective_iff_ker_eq_bot _).mpr h)


theorem ker_algebraMap_eq_bot (R A : Type*) [CommRing R] [Ring A] [Nontrivial A] [Algebra R A]
    [NoZeroSMulDivisors R A] : RingHom.ker (algebraMap R A) = ⊥ :=
  (RingHom.injective_iff_ker_eq_bot _).mp (algebraMap_injective R A)


theorem iff_ker_algebraMap_eq_bot {R A : Type*} [CommRing R] [Ring A] [IsDomain A] [Algebra R A] :
    NoZeroSMulDivisors R A ↔ RingHom.ker (algebraMap R A) = ⊥ :=
  iff_algebraMap_injective.trans (RingHom.injective_iff_ker_eq_bot (algebraMap R A))


