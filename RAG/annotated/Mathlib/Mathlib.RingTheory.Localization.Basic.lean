/-- `IsLocalization.map` applied to a projection homomorphism from a product ring. -/
noncomputable abbrev mapPiEvalRingHom :
    Localization (S.comap <| Pi.evalRingHom R i) →+* Localization S :=
  map (T := S) _ (Pi.evalRingHom R i) le_rfl


open Function in
theorem mapPiEvalRingHom_bijective : Bijective (mapPiEvalRingHom S) := by
  /-
    ι : Type u_1
    R : ι → Type u_2
    inst✝ : (i : ι) → CommSemiring (R i)
    i : ι
    S : Submonoid (R i)
    ⊢ Function.Bijective ⇑(Localization.mapPiEvalRingHom S)
  -/
  let T := S.comap (Pi.evalRingHom R i)
  classical
  refine ⟨fun x₁ x₂ eq ↦ ?_, fun x ↦ ?_⟩
  · obtain ⟨r₁, s₁, rfl⟩ := mk'_surjective T x₁
    obtain ⟨r₂, s₂, rfl⟩ := mk'_surjective T x₂
    simp_rw [map_mk'] at eq
    rw [IsLocalization.eq] at eq ⊢
    obtain ⟨s, hs⟩ := eq
    refine ⟨⟨update 0 i s, by apply update_self i s.1 0 ▸ s.2⟩, funext fun j ↦ ?_⟩
    obtain rfl | ne := eq_or_ne j i
    · simpa using hs
    · simp [update_of_ne ne]
  · obtain ⟨r, s, rfl⟩ := mk'_surjective S x
    exact ⟨mk' (M := T) _ (update 0 i r) ⟨update 0 i s, by apply update_self i s.1 0 ▸ s.2⟩,
      by simp [map_mk']⟩


variable (M S) in
include M in
theorem linearMap_compatibleSMul (N₁ N₂) [AddCommMonoid N₁] [AddCommMonoid N₂] [Module R N₁]
    [Module S N₁] [Module R N₂] [Module S N₂] [IsScalarTower R S N₁] [IsScalarTower R S N₂] :
    LinearMap.CompatibleSMul N₁ N₂ S R where
  map_smul f s s' := by
    /-
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹⁰ : CommSemiring S
      inst✝⁹ : Algebra R S
      inst✝⁸ : IsLocalization M S
      N₁ : Type u_4
      N₂ : Type u_5
      inst✝⁷ : AddCommMonoid N₁
      inst✝⁶ : AddCommMonoid N₂
      inst✝⁵ : Module R N₁
      inst✝⁴ : Module S N₁
      inst✝³ : Module R N₂
      inst✝² : Module S N₂
      inst✝¹ : IsScalarTower R S N₁
      inst✝ : IsScalarTower R S N₂
      f : LinearMap (RingHom.id R) N₁ N₂
      s : S
      s' : N₁
      ⊢ Eq (f (HSMul.hSMul s s')) (HSMul.hSMul s (f s'))
    -/
    obtain ⟨r, m, rfl⟩ := mk'_surjective M s
    /-
      case intro.intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹⁰ : CommSemiring S
      inst✝⁹ : Algebra R S
      inst✝⁸ : IsLocalization M S
      N₁ : Type u_4
      N₂ : Type u_5
      inst✝⁷ : AddCommMonoid N₁
      inst✝⁶ : AddCommMonoid N₂
      inst✝⁵ : Module R N₁
      inst✝⁴ : Module S N₁
      inst✝³ : Module R N₂
      inst✝² : Module S N₂
      inst✝¹ : IsScalarTower R S N₁
      inst✝ : IsScalarTower R S N₂
      f : LinearMap (RingHom.id R) N₁ N₂
      s' : N₁
      r : R
      m : Subtype fun x => Membership.mem M x
      ⊢ Eq (f (HSMul.hSMul (IsLocalization.mk' S r m) s')) (HSMul.hSMul (IsLocalizat …
    -/
    rw [← (map_units S m).smul_left_cancel]
    /-
      case intro.intro
      R : Type u_1
      inst✝¹¹ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹⁰ : CommSemiring S
      inst✝⁹ : Algebra R S
      inst✝⁸ : IsLocalization M S
      N₁ : Type u_4
      N₂ : Type u_5
      inst✝⁷ : AddCommMonoid N₁
      inst✝⁶ : AddCommMonoid N₂
      inst✝⁵ : Module R N₁
      inst✝⁴ : Module S N₁
      inst✝³ : Module R N₂
      inst✝² : Module S N₂
      inst✝¹ : IsScalarTower R S N₁
      inst✝ : IsScalarTower R S N₂
      f : LinearMap (RingHom.id R) N₁ N₂
      s' : N₁
      r : R
      m : Subtype fun x => Membership.mem M x
      ⊢ Eq (HSMul.hSMul ((algebraMap R S) ↑m) (f (HSMul.hSMul (IsLocalization.mk' S  …
    -/
    simp_rw [algebraMap_smul, ← map_smul, ← smul_assoc, smul_mk'_self, algebraMap_smul, map_smul]
    /-
      🎉 no goals
    -/


variable (M) in
include M in
/- This is not an instance because the submonoid `M` would become a metavariable
  in typeclass search. -/
theorem algHom_subsingleton [Algebra R P] : Subsingleton (S →ₐ[R] P) :=
  ⟨fun f g =>
    AlgHom.coe_ringHom_injective <|
                                         /-
                                           R : Type u_1
                                           inst✝⁵ : CommSemiring R
                                           M : Submonoid R
                                           S : Type u_2
                                           inst✝⁴ : CommSemiring S
                                           inst✝³ : Algebra R S
                                           P : Type u_3
                                           inst✝² : CommSemiring P
                                           inst✝¹ : IsLocalization M S
                                           inst✝ : Algebra R P
                                           f g : AlgHom R S P
                                           ⊢ Eq ((↑f).comp (algebraMap R S)) ((↑g).comp (algebraMap R S))
                                         -/
      IsLocalization.ringHom_ext M <| by rw [f.comp_algebraMap, g.comp_algebraMap]⟩
                                         /-
                                           🎉 no goals
                                         -/


/-- If `S`, `Q` are localizations of `R` at the submonoid `M` respectively,
there is an isomorphism of localizations `S ≃ₐ[R] Q`. -/
@[simps!]
noncomputable def algEquiv : S ≃ₐ[R] Q :=
  { ringEquivOfRingEquiv S Q (RingEquiv.refl R) M.map_id with
    commutes' := ringEquivOfRingEquiv_eq _ }


theorem algEquiv_mk' (x : R) (y : M) : algEquiv M S Q (mk' S x y) = mk' Q x y := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : IsLocalization M S
    Q : Type u_4
    inst✝² : CommSemiring Q
    inst✝¹ : Algebra R Q
    inst✝ : IsLocalization M Q
    x : R
    y : Subtype fun x => Membership.mem M x
    ⊢ Eq ((IsLocalization.algEquiv M S Q) (IsLocalization.mk' S x y)) (IsLocalizat …
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                                /-
                                                                                                  R : Type u_1
                                                                                                  inst✝⁶ : CommSemiring R
                                                                                                  M : Submonoid R
                                                                                                  S : Type u_2
                                                                                                  inst✝⁵ : CommSemiring S
                                                                                                  inst✝⁴ : Algebra R S
                                                                                                  inst✝³ : IsLocalization M S
                                                                                                  Q : Type u_4
                                                                                                  inst✝² : CommSemiring Q
                                                                                                  inst✝¹ : Algebra R Q
                                                                                                  inst✝ : IsLocalization M Q
                                                                                                  x : R
                                                                                                  y : Subtype fun x => Membership.mem M x
                                                                                                  ⊢ Eq ((IsLocalization.algEquiv M S Q).symm (IsLocalization.mk' Q x y)) (IsLoca …
                                                                                                -/
theorem algEquiv_symm_mk' (x : R) (y : M) : (algEquiv M S Q).symm (mk' Q x y) = mk' S x y := by simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


variable (M) in
include M in
protected lemma bijective (f : S →+* Q) (hf : f.comp (algebraMap R S) = algebraMap R Q) :
    Function.Bijective f :=
  (show f = IsLocalization.algEquiv M S Q by
    /-
      R : Type u_1
      inst✝⁶ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : IsLocalization M S
      Q : Type u_4
      inst✝² : CommSemiring Q
      inst✝¹ : Algebra R Q
      inst✝ : IsLocalization M Q
      f : RingHom S Q
      hf : Eq (f.comp (algebraMap R S)) (algebraMap R Q)
      ⊢ Eq f ↑(IsLocalization.algEquiv M S Q)
    -/
    apply IsLocalization.ringHom_ext M; rw [hf]; ext; simp) ▸
                                                      /-
                                                        🎉 no goals
                                                      -/
    (IsLocalization.algEquiv M S Q).toEquiv.bijective


/-- `AlgHom` version of `IsLocalization.lift`. -/
noncomputable def liftAlgHom : S →ₐ[A] P where
  __ := lift hf
  commutes' r := show lift hf (algebraMap A S r) = _ by
    /-
      R✝ : Type u_1
      inst✝¹⁴ : CommSemiring R✝
      M✝ N : Submonoid R✝
      S✝ : Type u_2
      inst✝¹³ : CommSemiring S✝
      inst✝¹² : Algebra R✝ S✝
      P✝ : Type u_3
      inst✝¹¹ : CommSemiring P✝
      inst✝¹⁰ : IsLocalization M✝ S✝
      g : RingHom R✝ P✝
      hg : ∀ (y : Subtype fun x => Membership.mem M✝ x), IsUnit (g ↑y)
      A : Type u_4
      inst✝⁹ : CommSemiring A
      R : Type u_5
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Algebra A R
      M : Submonoid R
      S : Type u_6
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra A S
      inst✝⁴ : Algebra R S
      inst✝³ : IsScalarTower A R S
      P : Type u_7
      inst✝² : CommSemiring P
      inst✝¹ : Algebra A P
      inst✝ : IsLocalization M S
      f : AlgHom A R P
      hf : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (f ↑y)
      x : S
      r : A
      ⊢ Eq ((IsLocalization.lift hf) ((algebraMap A S) r)) ((algebraMap A P) r)
    -/
    simp [IsScalarTower.algebraMap_apply A R S]
    /-
      🎉 no goals
    -/


theorem liftAlgHom_toRingHom : (liftAlgHom hf : S →ₐ[A] P).toRingHom = lift hf := rfl


@[simp]
theorem coe_liftAlgHom : ⇑(liftAlgHom hf : S →ₐ[A] P) = lift hf := rfl


theorem liftAlgHom_apply : liftAlgHom hf x = lift hf x := rfl


/-- If `S`, `Q` are localizations of `R` and `P` at submonoids `M`, `T` respectively,
an isomorphism `h : R ≃ₐ[A] P` such that `h(M) = T` induces an isomorphism of localizations
`S ≃ₐ[A] Q`. -/
@[simps!]
noncomputable def algEquivOfAlgEquiv : S ≃ₐ[A] Q where
  __ := ringEquivOfRingEquiv S Q h.toRingEquiv H
                    /-
                      R✝ : Type u_1
                      inst✝¹⁹ : CommSemiring R✝
                      M✝ N : Submonoid R✝
                      S✝ : Type u_2
                      inst✝¹⁸ : CommSemiring S✝
                      inst✝¹⁷ : Algebra R✝ S✝
                      P✝ : Type u_3
                      inst✝¹⁶ : CommSemiring P✝
                      inst✝¹⁵ : IsLocalization M✝ S✝
                      g : RingHom R✝ P✝
                      hg : ∀ (y : Subtype fun x => Membership.mem M✝ x), IsUnit (g ↑y)
                      A : Type u_4
                      inst✝¹⁴ : CommSemiring A
                      R : Type u_5
                      inst✝¹³ : CommSemiring R
                      inst✝¹² : Algebra A R
                      M : Submonoid R
                      S : Type u_6
                      inst✝¹¹ : CommSemiring S
                      inst✝¹⁰ : Algebra A S
                      inst✝⁹ : Algebra R S
                      inst✝⁸ : IsScalarTower A R S
                      inst✝⁷ : IsLocalization M S
                      P : Type u_7
                      inst✝⁶ : CommSemiring P
                      inst✝⁵ : Algebra A P
                      T : Submonoid P
                      Q : Type u_8
                      inst✝⁴ : CommSemiring Q
                      inst✝³ : Algebra A Q
                      inst✝² : Algebra P Q
                      inst✝¹ : IsScalarTower A P Q
                      inst✝ : IsLocalization T Q
                      h : AlgEquiv A R P
                      H : Eq (Submonoid.map h M) T
                      x✝ : A
                      ⊢ Eq (__spread✝⁻⁰.toFun ((algebraMap A S) x✝)) ((algebraMap A Q) x✝)
                    -/
  commutes' _ := by dsimp; rw [IsScalarTower.algebraMap_apply A R S, map_eq,
    RingHom.coe_coe, AlgEquiv.commutes, IsScalarTower.algebraMap_apply A P Q]


theorem algEquivOfAlgEquiv_eq_map :
    (algEquivOfAlgEquiv S Q h H : S →+* Q) =
      map Q (h : R →+* P) (M.le_comap_of_map_le (le_of_eq H)) :=
  rfl


theorem algEquivOfAlgEquiv_eq (x : R) :
    algEquivOfAlgEquiv S Q h H ((algebraMap R S) x) = algebraMap P Q (h x) := by
  /-
    A : Type u_4
    inst✝¹⁴ : CommSemiring A
    R : Type u_5
    inst✝¹³ : CommSemiring R
    inst✝¹² : Algebra A R
    M : Submonoid R
    S : Type u_6
    inst✝¹¹ : CommSemiring S
    inst✝¹⁰ : Algebra A S
    inst✝⁹ : Algebra R S
    inst✝⁸ : IsScalarTower A R S
    inst✝⁷ : IsLocalization M S
    P : Type u_7
    inst✝⁶ : CommSemiring P
    inst✝⁵ : Algebra A P
    T : Submonoid P
    Q : Type u_8
    inst✝⁴ : CommSemiring Q
    inst✝³ : Algebra A Q
    inst✝² : Algebra P Q
    inst✝¹ : IsScalarTower A P Q
    inst✝ : IsLocalization T Q
    h : AlgEquiv A R P
    H : Eq (Submonoid.map h M) T
    x : R
    ⊢ Eq ((IsLocalization.algEquivOfAlgEquiv S Q h H) ((algebraMap R S) x)) ((alge …
  -/
  simp
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
theorem algEquivOfAlgEquiv_mk' (x : R) (y : M) :
    algEquivOfAlgEquiv S Q h H (mk' S x y) =
      mk' Q (h x) ⟨h y, show h y ∈ T from H ▸ Set.mem_image_of_mem h y.2⟩ := by
  /-
    A : Type u_4
    inst✝¹⁴ : CommSemiring A
    R : Type u_5
    inst✝¹³ : CommSemiring R
    inst✝¹² : Algebra A R
    M : Submonoid R
    S : Type u_6
    inst✝¹¹ : CommSemiring S
    inst✝¹⁰ : Algebra A S
    inst✝⁹ : Algebra R S
    inst✝⁸ : IsScalarTower A R S
    inst✝⁷ : IsLocalization M S
    P : Type u_7
    inst✝⁶ : CommSemiring P
    inst✝⁵ : Algebra A P
    T : Submonoid P
    Q : Type u_8
    inst✝⁴ : CommSemiring Q
    inst✝³ : Algebra A Q
    inst✝² : Algebra P Q
    inst✝¹ : IsScalarTower A P Q
    inst✝ : IsLocalization T Q
    h : AlgEquiv A R P
    H : Eq (Submonoid.map h M) T
    x : R
    y : Subtype fun x => Membership.mem M x
    ⊢ Eq ((IsLocalization.algEquivOfAlgEquiv S Q h H) (IsLocalization.mk' S x y))  …
  -/
  simp [map_mk']
  /-
    🎉 no goals
  -/


theorem algEquivOfAlgEquiv_symm : (algEquivOfAlgEquiv S Q h H).symm =
    algEquivOfAlgEquiv Q S h.symm (show Submonoid.map h.symm T = M by
      erw [← H, ← Submonoid.comap_equiv_eq_map_symm,
        Submonoid.comap_map_eq_of_injective h.injective]) := rfl


/-- The localization at a module of units is isomorphic to the ring. -/
noncomputable def atUnits (H : M ≤ IsUnit.submonoid R) : R ≃ₐ[R] S := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M N : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    P : Type u_3
    inst✝¹ : CommSemiring P
    inst✝ : IsLocalization M S
    g : RingHom R P
    hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
    H : LE.le M (IsUnit.submonoid R)
    ⊢ AlgEquiv R R S
  -/
  refine AlgEquiv.ofBijective (Algebra.ofId R S) ⟨?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      ⊢ Function.Injective ⇑(Algebra.ofId R S)
    -/
  · intro x y hxy
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      x y : R
      hxy : Eq ((Algebra.ofId R S) x) ((Algebra.ofId R S) y)
      ⊢ Eq x y
    -/
    obtain ⟨c, eq⟩ := (IsLocalization.eq_iff_exists M S).mp hxy
    /-
      case refine_1.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      x y : R
      hxy : Eq ((Algebra.ofId R S) x) ((Algebra.ofId R S) y)
      c : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      ⊢ Eq x y
    -/
    obtain ⟨u, hu⟩ := H c.prop
    /-
      case refine_1.intro.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      x y : R
      hxy : Eq ((Algebra.ofId R S) x) ((Algebra.ofId R S) y)
      c : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      u : Units R
      hu : Eq ↑u ↑c
      ⊢ Eq x y
    -/
    rwa [← hu, Units.mul_right_inj] at eq
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      ⊢ Function.Surjective ⇑(Algebra.ofId R S)
    -/
  · intro y
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      y : S
      ⊢ Exists fun a => Eq ((Algebra.ofId R S) a) y
    -/
    obtain ⟨⟨x, s⟩, eq⟩ := IsLocalization.surj M y
    /-
      case refine_2.intro.mk
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMa …
      ⊢ Exists fun a => Eq ((Algebra.ofId R S) a) y
    -/
    obtain ⟨u, hu⟩ := H s.prop
    /-
      case refine_2.intro.mk.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMa …
      u : Units R
      hu : Eq ↑u ↑s
      ⊢ Exists fun a => Eq ((Algebra.ofId R S) a) y
    -/
    use x * u.inv
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMa …
      u : Units R
      hu : Eq ↑u ↑s
      ⊢ Eq ((Algebra.ofId R S) (HMul.hMul x u.inv)) y
    -/
    dsimp [Algebra.ofId, RingHom.toFun_eq_coe, AlgHom.coe_mks]
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMa …
      u : Units R
      hu : Eq ↑u ↑s
      ⊢ Eq ((algebraMap R S) (HMul.hMul x ↑(Inv.inv u))) y
    -/
    rw [RingHom.map_mul, ← eq, ← hu, mul_assoc, ← RingHom.map_mul]
    /-
      case h
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M N : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      H : LE.le M (IsUnit.submonoid R)
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      eq : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMa …
      u : Units R
      hu : Eq ↑u ↑s
      ⊢ Eq (HMul.hMul y ((algebraMap R S) (HMul.hMul ↑u ↑(Inv.inv u)))) y
    -/
    simp
    /-
      🎉 no goals
    -/


theorem isLocalization_of_algEquiv [Algebra R P] [IsLocalization M S] (h : S ≃ₐ[R] P) :
    IsLocalization M P := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁴ : CommSemiring S
    inst✝³ : Algebra R S
    P : Type u_3
    inst✝² : CommSemiring P
    inst✝¹ : Algebra R P
    inst✝ : IsLocalization M S
    h : AlgEquiv R S P
    ⊢ IsLocalization M P
  -/
  constructor
    /-
      case map_units'
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      ⊢ ∀ (y : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R P) ↑y)
    -/
  · intro y
    /-
      case map_units'
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      y : Subtype fun x => Membership.mem M x
      ⊢ IsUnit ((algebraMap R P) ↑y)
    -/
    convert (IsLocalization.map_units S y).map h.toAlgHom.toRingHom.toMonoidHom
    /-
      case h.e'_3
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      y : Subtype fun x => Membership.mem M x
      ⊢ Eq ((algebraMap R P) ↑y) (↑(↑h).toRingHom ((algebraMap R S) ↑y))
    -/
    exact (h.commutes y).symm
    /-
      🎉 no goals
    -/
    /-
      case surj'
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      ⊢ ∀ (z : P), Exists fun x => Eq (HMul.hMul z ((algebraMap R P) ↑x.2)) ((algebr …
    -/
  · intro y
    /-
      case surj'
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      y : P
      ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap R P) ↑x.2)) ((algebraMap R P) x …
    -/
    obtain ⟨⟨x, s⟩, e⟩ := IsLocalization.surj M (h.symm y)
    /-
      case surj'.intro.mk
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      y : P
      x : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul (h.symm y) ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((a …
      ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap R P) ↑x.2)) ((algebraMap R P) x …
    -/
    apply_fun (show S → P from h) at e
    /-
      case surj'.intro.mk
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      y : P
      x : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (letFun (⇑h) (fun this => this) (HMul.hMul (h.symm y) ((algebraMap R S) …
      ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap R P) ↑x.2)) ((algebraMap R P) x …
    -/
    simp only [map_mul, h.apply_symm_apply, h.commutes] at e
    /-
      case surj'.intro.mk
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      y : P
      x : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul y ((algebraMap R P) ↑s)) ((algebraMap R P) x)
      ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap R P) ↑x.2)) ((algebraMap R P) x …
    -/
    exact ⟨⟨x, s⟩, e⟩
    /-
      🎉 no goals
    -/
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      ⊢ ∀ {x y : R}, Eq ((algebraMap R P) x) ((algebraMap R P) y) → Exists fun c =>  …
    -/
  · intro x y
    rw [← h.symm.toEquiv.injective.eq_iff, ← IsLocalization.eq_iff_exists M S, ← h.symm.commutes, ←
      h.symm.commutes]
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommSemiring S
      inst✝³ : Algebra R S
      P : Type u_3
      inst✝² : CommSemiring P
      inst✝¹ : Algebra R P
      inst✝ : IsLocalization M S
      h : AlgEquiv R S P
      x y : R
      ⊢ Eq (h.symm.toEquiv ((algebraMap R P) x)) (h.symm.toEquiv ((algebraMap R P) y …
    -/
    exact id
    /-
      🎉 no goals
    -/


theorem isLocalization_iff_of_algEquiv [Algebra R P] (h : S ≃ₐ[R] P) :
    IsLocalization M S ↔ IsLocalization M P :=
  ⟨fun _ => isLocalization_of_algEquiv M h, fun _ => isLocalization_of_algEquiv M h.symm⟩


theorem isLocalization_iff_of_ringEquiv (h : S ≃+* P) :
    IsLocalization M S ↔
      haveI := (h.toRingHom.comp <| algebraMap R S).toAlgebra; IsLocalization M P :=
  letI := (h.toRingHom.comp <| algebraMap R S).toAlgebra
  isLocalization_iff_of_algEquiv M { h with commutes' := fun _ => rfl }


variable (S) in
/-- If an algebra is simultaneously localizations for two submonoids, then an arbitrary algebra
is a localization of one submonoid iff it is a localization of the other. -/
theorem isLocalization_iff_of_isLocalization [IsLocalization M S] [IsLocalization N S]
    [Algebra R P] : IsLocalization M P ↔ IsLocalization N P :=
  ⟨fun _ ↦ isLocalization_of_algEquiv N (algEquiv M S P),
    fun _ ↦ isLocalization_of_algEquiv M (algEquiv N S P)⟩


/-- If `S₁` is the localization of `R` at `M₁` and `S₂` is the localization of
`R` at `M₂`, then every localization `T` of `S₂` at `M₁` is also a localization of
`S₁` at `M₂`, in other words `M₁⁻¹M₂⁻¹R` can be identified with `M₂⁻¹M₁⁻¹R`. -/
lemma commutes (S₁ S₂ T : Type*) [CommSemiring S₁]
    [CommSemiring S₂] [CommSemiring T] [Algebra R S₁] [Algebra R S₂] [Algebra R T] [Algebra S₁ T]
    [Algebra S₂ T] [IsScalarTower R S₁ T] [IsScalarTower R S₂ T] (M₁ M₂ : Submonoid R)
    [IsLocalization M₁ S₁] [IsLocalization M₂ S₂]
    [IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T] :
    IsLocalization (Algebra.algebraMapSubmonoid S₁ M₂) T where
  map_units' := by
    /-
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Algebra.algebraMapSubmonoid S₁ M₂) x …
    -/
    rintro ⟨m, ⟨a, ha, rfl⟩⟩
    /-
      case mk.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : R
      ha : Membership.mem (↑M₂) a
      ⊢ IsUnit ((algebraMap S₁ T) ↑⟨(algebraMap R S₁) a, ⋯⟩)
    -/
    rw [← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply R S₂ T]
    /-
      case mk.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : R
      ha : Membership.mem (↑M₂) a
      ⊢ IsUnit ((algebraMap S₂ T) ((algebraMap R S₂) a))
    -/
    exact IsUnit.map _ (IsLocalization.map_units' ⟨a, ha⟩)
    /-
      🎉 no goals
    -/
  surj' a := by
    /-
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      ⊢ Exists fun x => Eq (HMul.hMul a ((algebraMap S₁ T) ↑x.2)) ((algebraMap S₁ T) …
    -/
    obtain ⟨⟨y, -, m, hm, rfl⟩, hy⟩ := surj (M := Algebra.algebraMapSubmonoid S₂ M₁) a
    /-
      case intro.mk.mk.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₂ T) ↑{ fst := y, snd := ⟨(algebraMap R S₂) …
      ⊢ Exists fun x => Eq (HMul.hMul a ((algebraMap S₁ T) ↑x.2)) ((algebraMap S₁ T) …
    -/
    rw [← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply R S₁ T] at hy
    /-
      case intro.mk.mk.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      ⊢ Exists fun x => Eq (HMul.hMul a ((algebraMap S₁ T) ↑x.2)) ((algebraMap S₁ T) …
    -/
    obtain ⟨⟨z, n, hn⟩, hz⟩ := IsLocalization.surj (M := M₂) y
    /-
      case intro.mk.mk.intro.intro.intro.mk.mk
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      z n : R
      hn : Membership.mem M₂ n
      hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
      ⊢ Exists fun x => Eq (HMul.hMul a ((algebraMap S₁ T) ↑x.2)) ((algebraMap S₁ T) …
    -/
    have hunit : IsUnit (algebraMap R S₁ m) := map_units' ⟨m, hm⟩
    /-
      case intro.mk.mk.intro.intro.intro.mk.mk
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      z n : R
      hn : Membership.mem M₂ n
      hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
      hunit : IsUnit ((algebraMap R S₁) m)
      ⊢ Exists fun x => Eq (HMul.hMul a ((algebraMap S₁ T) ↑x.2)) ((algebraMap S₁ T) …
    -/
    use ⟨algebraMap R S₁ z * hunit.unit⁻¹, ⟨algebraMap R S₁ n, n, hn, rfl⟩⟩
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      z n : R
      hn : Membership.mem M₂ n
      hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
      hunit : IsUnit ((algebraMap R S₁) m)
      ⊢ Eq (HMul.hMul a ((algebraMap S₁ T) ↑{ fst := HMul.hMul ((algebraMap R S₁) z) …
    -/
    rw [map_mul, ← IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply R S₂ T]
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      z n : R
      hn : Membership.mem M₂ n
      hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
      hunit : IsUnit ((algebraMap R S₁) m)
      ⊢ Eq (HMul.hMul a ((algebraMap S₂ T) ((algebraMap R S₂) n))) (HMul.hMul ((alge …
    -/
    conv_rhs => rw [← IsScalarTower.algebraMap_apply]
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      z n : R
      hn : Membership.mem M₂ n
      hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
      hunit : IsUnit ((algebraMap R S₁) m)
      ⊢ Eq (HMul.hMul a ((algebraMap S₂ T) ((algebraMap R S₂) n))) (HMul.hMul ((alge …
    -/
    rw [IsScalarTower.algebraMap_apply R S₂ T, ← hz, map_mul, ← hy]
    convert_to _ = a * (algebraMap S₂ T) ((algebraMap R S₂) n) *
        (algebraMap S₁ T) (((algebraMap R S₁) m) * hunit.unit⁻¹.val)
      /-
        case h.e'_3
        R : Type u_1
        inst✝¹³ : CommSemiring R
        S₁ : Type u_4
        S₂ : Type u_5
        T : Type u_6
        inst✝¹² : CommSemiring S₁
        inst✝¹¹ : CommSemiring S₂
        inst✝¹⁰ : CommSemiring T
        inst✝⁹ : Algebra R S₁
        inst✝⁸ : Algebra R S₂
        inst✝⁷ : Algebra R T
        inst✝⁶ : Algebra S₁ T
        inst✝⁵ : Algebra S₂ T
        inst✝⁴ : IsScalarTower R S₁ T
        inst✝³ : IsScalarTower R S₂ T
        M₁ M₂ : Submonoid R
        inst✝² : IsLocalization M₁ S₁
        inst✝¹ : IsLocalization M₂ S₂
        inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
        a : T
        y : S₂
        m : R
        hm : Membership.mem (↑M₁) m
        hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
        z n : R
        hn : Membership.mem M₂ n
        hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
        hunit : IsUnit ((algebraMap R S₁) m)
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁)  …
      -/
    · rw [map_mul]
      /-
        case h.e'_3
        R : Type u_1
        inst✝¹³ : CommSemiring R
        S₁ : Type u_4
        S₂ : Type u_5
        T : Type u_6
        inst✝¹² : CommSemiring S₁
        inst✝¹¹ : CommSemiring S₂
        inst✝¹⁰ : CommSemiring T
        inst✝⁹ : Algebra R S₁
        inst✝⁸ : Algebra R S₂
        inst✝⁷ : Algebra R T
        inst✝⁶ : Algebra S₁ T
        inst✝⁵ : Algebra S₂ T
        inst✝⁴ : IsScalarTower R S₁ T
        inst✝³ : IsScalarTower R S₂ T
        M₁ M₂ : Submonoid R
        inst✝² : IsLocalization M₁ S₁
        inst✝¹ : IsLocalization M₂ S₂
        inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
        a : T
        y : S₂
        m : R
        hm : Membership.mem (↑M₁) m
        hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
        z n : R
        hn : Membership.mem M₂ n
        hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
        hunit : IsUnit ((algebraMap R S₁) m)
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁)  …
      -/
      ring
      /-
        🎉 no goals
      -/
    /-
      case h.convert_2
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      a : T
      y : S₂
      m : R
      hm : Membership.mem (↑M₁) m
      hy : Eq (HMul.hMul a ((algebraMap S₁ T) ((algebraMap R S₁) m))) ((algebraMap S …
      z n : R
      hn : Membership.mem M₂ n
      hz : Eq (HMul.hMul y ((algebraMap R S₂) ↑{ fst := z, snd := ⟨n, hn⟩ }.2)) ((al …
      hunit : IsUnit ((algebraMap R S₁) m)
      ⊢ Eq (HMul.hMul a ((algebraMap S₂ T) ((algebraMap R S₂) n))) (HMul.hMul (HMul. …
    -/
    simp
    /-
      🎉 no goals
    -/
  exists_of_eq {x y} hxy := by
    /-
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      hxy : Eq ((algebraMap S₁ T) x) ((algebraMap S₁ T) y)
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    obtain ⟨r, s, d, hr, hs⟩ := IsLocalization.surj₂ M₁ S₁ x y
    /-
      case intro.intro.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      hxy : Eq ((algebraMap S₁ T) x) ((algebraMap S₁ T) y)
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    apply_fun (· * algebraMap S₁ T (algebraMap R S₁ d)) at hxy
    simp_rw [← map_mul, hr, hs, ← IsScalarTower.algebraMap_apply,
      IsScalarTower.algebraMap_apply R S₂ T] at hxy
    /-
      case intro.intro.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    obtain ⟨⟨-, c, hmc, rfl⟩, hc⟩ := exists_of_eq (M := Algebra.algebraMapSubmonoid S₂ M₁) hxy
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq (HMul.hMul (↑⟨(algebraMap R S₂) c, ⋯⟩) ((algebraMap R S₂) r)) (HMul.hM …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    simp_rw [← map_mul] at hc
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    obtain ⟨a, ha⟩ := IsLocalization.exists_of_eq (M := M₂) hc
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro.intro
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      a : Subtype fun x => Membership.mem M₂ x
      ha : Eq (HMul.hMul (↑a) (HMul.hMul c r)) (HMul.hMul (↑a) (HMul.hMul c s))
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    use ⟨algebraMap R S₁ a, a, a.property, rfl⟩
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      a : Subtype fun x => Membership.mem M₂ x
      ha : Eq (HMul.hMul (↑a) (HMul.hMul c r)) (HMul.hMul (↑a) (HMul.hMul c s))
      ⊢ Eq (HMul.hMul (↑⟨(algebraMap R S₁) ↑a, ⋯⟩) x) (HMul.hMul (↑⟨(algebraMap R S₁ …
    -/
    apply (map_units S₁ d).mul_right_cancel
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      a : Subtype fun x => Membership.mem M₂ x
      ha : Eq (HMul.hMul (↑a) (HMul.hMul c r)) (HMul.hMul (↑a) (HMul.hMul c s))
      ⊢ Eq (HMul.hMul (HMul.hMul (↑⟨(algebraMap R S₁) ↑a, ⋯⟩) x) ((algebraMap R S₁)  …
    -/
    rw [mul_assoc, hr, mul_assoc, hs]
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      a : Subtype fun x => Membership.mem M₂ x
      ha : Eq (HMul.hMul (↑a) (HMul.hMul c r)) (HMul.hMul (↑a) (HMul.hMul c s))
      ⊢ Eq (HMul.hMul (↑⟨(algebraMap R S₁) ↑a, ⋯⟩) ((algebraMap R S₁) r)) (HMul.hMul …
    -/
    apply (map_units S₁ ⟨c, hmc⟩).mul_right_cancel
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      a : Subtype fun x => Membership.mem M₂ x
      ha : Eq (HMul.hMul (↑a) (HMul.hMul c r)) (HMul.hMul (↑a) (HMul.hMul c s))
      ⊢ Eq (HMul.hMul (HMul.hMul (↑⟨(algebraMap R S₁) ↑a, ⋯⟩) ((algebraMap R S₁) r)) …
    -/
    rw [← map_mul, ← map_mul, mul_assoc, mul_comm _ c, ha, map_mul, map_mul]
    /-
      case h
      R : Type u_1
      inst✝¹³ : CommSemiring R
      S₁ : Type u_4
      S₂ : Type u_5
      T : Type u_6
      inst✝¹² : CommSemiring S₁
      inst✝¹¹ : CommSemiring S₂
      inst✝¹⁰ : CommSemiring T
      inst✝⁹ : Algebra R S₁
      inst✝⁸ : Algebra R S₂
      inst✝⁷ : Algebra R T
      inst✝⁶ : Algebra S₁ T
      inst✝⁵ : Algebra S₂ T
      inst✝⁴ : IsScalarTower R S₁ T
      inst✝³ : IsScalarTower R S₂ T
      M₁ M₂ : Submonoid R
      inst✝² : IsLocalization M₁ S₁
      inst✝¹ : IsLocalization M₂ S₂
      inst✝ : IsLocalization (Algebra.algebraMapSubmonoid S₂ M₁) T
      x y : S₁
      r s : R
      d : Subtype fun x => Membership.mem M₁ x
      hr : Eq (HMul.hMul x ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) r)
      hs : Eq (HMul.hMul y ((algebraMap R S₁) ↑d)) ((algebraMap R S₁) s)
      hxy : Eq ((algebraMap S₂ T) ((algebraMap R S₂) r)) ((algebraMap S₂ T) ((algebr …
      c : R
      hmc : Membership.mem (↑M₁) c
      hc : Eq ((algebraMap R S₂) (HMul.hMul c r)) ((algebraMap R S₂) (HMul.hMul c s))
      a : Subtype fun x => Membership.mem M₂ x
      ha : Eq (HMul.hMul (↑a) (HMul.hMul c r)) (HMul.hMul (↑a) (HMul.hMul c s))
      ⊢ Eq (HMul.hMul ((algebraMap R S₁) ↑a) (HMul.hMul ((algebraMap R S₁) c) ((alge …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem mk_natCast (m : ℕ) : (mk m 1 : Localization M) = m := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    m : Nat
    ⊢ Eq (Localization.mk (↑m) 1) ↑m
  -/
  simpa using mk_algebraMap (R := R) (A := ℕ) _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias mk_nat_cast := mk_natCast


/-- The localization of `R` at `M` as a quotient type is isomorphic to any other localization. -/
@[simps!]
noncomputable def algEquiv : Localization M ≃ₐ[R] S :=
  IsLocalization.algEquiv M _ _


/-- The localization of a singleton is a singleton. Cannot be an instance due to metavariables. -/
noncomputable def _root_.IsLocalization.unique (R Rₘ) [CommSemiring R] [CommSemiring Rₘ]
    (M : Submonoid R) [Subsingleton R] [Algebra R Rₘ] [IsLocalization M Rₘ] : Unique Rₘ :=
  have : Inhabited Rₘ := ⟨1⟩
  (algEquiv M Rₘ).symm.injective.unique


nonrec theorem algEquiv_mk' (x : R) (y : M) : algEquiv M S (mk' (Localization M) x y) = mk' S x y :=
  algEquiv_mk' _ _


nonrec theorem algEquiv_symm_mk' (x : R) (y : M) :
    (algEquiv M S).symm (mk' S x y) = mk' (Localization M) x y :=
  algEquiv_symm_mk' _ _


                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝³ : CommSemiring R
                                                                      M : Submonoid R
                                                                      S : Type u_2
                                                                      inst✝² : CommSemiring S
                                                                      inst✝¹ : Algebra R S
                                                                      inst✝ : IsLocalization M S
                                                                      x : R
                                                                      y : Subtype fun x => Membership.mem M x
                                                                      ⊢ Eq ((Localization.algEquiv M S) (Localization.mk x y)) (IsLocalization.mk' S …
                                                                    -/
theorem algEquiv_mk (x y) : algEquiv M S (mk x y) = mk' S x y := by rw [mk_eq_mk', algEquiv_mk']
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem algEquiv_symm_mk (x : R) (y : M) : (algEquiv M S).symm (mk' S x y) = mk x y := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    y : Subtype fun x => Membership.mem M x
    ⊢ Eq ((Localization.algEquiv M S).symm (IsLocalization.mk' S x y)) (Localizati …
  -/
  rw [mk_eq_mk', algEquiv_symm_mk']
  /-
    🎉 no goals
  -/


lemma coe_algEquiv :
    (Localization.algEquiv M S : Localization M →+* S) =
    IsLocalization.map (M := M) (T := M) _ (RingHom.id R) le_rfl := rfl


lemma coe_algEquiv_symm :
    ((Localization.algEquiv M S).symm : S →+* Localization M) =
    IsLocalization.map (M := M) (T := M) _ (RingHom.id R) le_rfl := rfl


theorem mk_intCast (m : ℤ) : (mk m 1 : Localization M) = m := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Submonoid R
    m : Int
    ⊢ Eq (Localization.mk (↑m) 1) ↑m
  -/
  simpa using mk_algebraMap (R := R) (A := ℤ) _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias mk_int_cast := mk_intCast


/-- If `R` is a field, then localizing at a submonoid not containing `0` adds no new elements. -/
theorem IsField.localization_map_bijective {R Rₘ : Type*} [CommRing R] [CommRing Rₘ]
    {M : Submonoid R} (hM : (0 : R) ∉ M) (hR : IsField R) [Algebra R Rₘ] [IsLocalization M Rₘ] :
    Function.Bijective (algebraMap R Rₘ) := by
  /-
    R : Type u_4
    Rₘ : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    hM : Not (Membership.mem M 0)
    hR : IsField R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    ⊢ Function.Bijective ⇑(algebraMap R Rₘ)
  -/
  letI := hR.toField
  /-
    R : Type u_4
    Rₘ : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    hM : Not (Membership.mem M 0)
    hR : IsField R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    this : Field R := hR.toField
    ⊢ Function.Bijective ⇑(algebraMap R Rₘ)
  -/
  replace hM := le_nonZeroDivisors_of_noZeroDivisors hM
  /-
    R : Type u_4
    Rₘ : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    hR : IsField R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    this : Field R := hR.toField
    hM : LE.le M (nonZeroDivisors R)
    ⊢ Function.Bijective ⇑(algebraMap R Rₘ)
  -/
  refine ⟨IsLocalization.injective _ hM, fun x => ?_⟩
  /-
    R : Type u_4
    Rₘ : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    hR : IsField R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    this : Field R := hR.toField
    hM : LE.le M (nonZeroDivisors R)
    x : Rₘ
    ⊢ Exists fun a => Eq ((algebraMap R Rₘ) a) x
  -/
  obtain ⟨r, ⟨m, hm⟩, rfl⟩ := mk'_surjective M x
  /-
    case intro.intro.mk
    R : Type u_4
    Rₘ : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    hR : IsField R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    this : Field R := hR.toField
    hM : LE.le M (nonZeroDivisors R)
    r m : R
    hm : Membership.mem M m
    ⊢ Exists fun a => Eq ((algebraMap R Rₘ) a) (IsLocalization.mk' Rₘ r ⟨m, hm⟩)
  -/
  obtain ⟨n, hn⟩ := hR.mul_inv_cancel (nonZeroDivisors.ne_zero <| hM hm)
  /-
    case intro.intro.mk.intro
    R : Type u_4
    Rₘ : Type u_5
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    hR : IsField R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    this : Field R := hR.toField
    hM : LE.le M (nonZeroDivisors R)
    r m : R
    hm : Membership.mem M m
    n : R
    hn : Eq (HMul.hMul m n) 1
    ⊢ Exists fun a => Eq ((algebraMap R Rₘ) a) (IsLocalization.mk' Rₘ r ⟨m, hm⟩)
  -/
  exact ⟨r * n, by rw [eq_mk'_iff_mul_eq, ← map_mul, mul_assoc, _root_.mul_comm n, hn, mul_one]⟩
  /-
    🎉 no goals
  -/


/-- If `R` is a field, then localizing at a submonoid not containing `0` adds no new elements. -/
theorem Field.localization_map_bijective {K Kₘ : Type*} [Field K] [CommRing Kₘ] {M : Submonoid K}
    (hM : (0 : K) ∉ M) [Algebra K Kₘ] [IsLocalization M Kₘ] :
    Function.Bijective (algebraMap K Kₘ) :=
  (Field.toIsField K).localization_map_bijective hM

-- this looks weird due to the `letI` inside the above lemma, but trying to do it the other
-- way round causes issues with defeq of instances, so this is actually easier.

/-- Definition of the natural algebra induced by the localization of an algebra.
Given an algebra `R → S`, a submonoid `R` of `M`, and a localization `Rₘ` for `M`,
let `Sₘ` be the localization of `S` to the image of `M` under `algebraMap R S`.
Then this is the natural algebra structure on `Rₘ → Sₘ`, such that the entire square commutes,
where `localization_map.map_comp` gives the commutativity of the underlying maps.

This instance can be helpful if you define `Sₘ := Localization (Algebra.algebraMapSubmonoid S M)`,
however we will instead use the hypotheses `[Algebra Rₘ Sₘ] [IsScalarTower R Rₘ Sₘ]` in lemmas
since the algebra structure may arise in different ways.
-/
noncomputable def localizationAlgebra : Algebra Rₘ Sₘ :=
  (map Sₘ (algebraMap R S)
        (show _ ≤ (Algebra.algebraMapSubmonoid S M).comap _ from M.le_comap_map) :
      Rₘ →+* Sₘ).toAlgebra


theorem IsLocalization.map_units_map_submonoid (y : M) : IsUnit (algebraMap R Sₘ y) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    Sₘ : Type u_5
    inst✝³ : CommRing Sₘ
    inst✝² : Algebra S Sₘ
    i : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝¹ : Algebra R Sₘ
    inst✝ : IsScalarTower R S Sₘ
    y : Subtype fun x => Membership.mem M x
    ⊢ IsUnit ((algebraMap R Sₘ) ↑y)
  -/
  rw [IsScalarTower.algebraMap_apply _ S]
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    Sₘ : Type u_5
    inst✝³ : CommRing Sₘ
    inst✝² : Algebra S Sₘ
    i : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝¹ : Algebra R Sₘ
    inst✝ : IsScalarTower R S Sₘ
    y : Subtype fun x => Membership.mem M x
    ⊢ IsUnit ((algebraMap S Sₘ) ((algebraMap R S) ↑y))
  -/
  exact IsLocalization.map_units Sₘ ⟨algebraMap R S y, Algebra.mem_algebraMapSubmonoid_of_mem y⟩
  /-
    🎉 no goals
  -/

-- can't be simp, as `S` only appears on the RHS

theorem IsLocalization.algebraMap_mk' (x : R) (y : M) :
    algebraMap Rₘ Sₘ (IsLocalization.mk' Rₘ x y) =
      IsLocalization.mk' Sₘ (algebraMap R S x)
        ⟨algebraMap R S y, Algebra.mem_algebraMapSubmonoid_of_mem y⟩ := by
  rw [IsLocalization.eq_mk'_iff_mul_eq, Subtype.coe_mk, ← IsScalarTower.algebraMap_apply, ←
    IsScalarTower.algebraMap_apply, IsScalarTower.algebraMap_apply R Rₘ Sₘ,
    IsScalarTower.algebraMap_apply R Rₘ Sₘ, ← _root_.map_mul, mul_comm,
    IsLocalization.mul_mk'_eq_mk'_of_mul]
  /-
    R : Type u_1
    inst✝¹¹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    Rₘ : Type u_4
    Sₘ : Type u_5
    inst✝⁸ : CommRing Rₘ
    inst✝⁷ : CommRing Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : IsLocalization M Rₘ
    inst✝⁴ : Algebra S Sₘ
    i : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝³ : Algebra Rₘ Sₘ
    inst✝² : Algebra R Sₘ
    inst✝¹ : IsScalarTower R Rₘ Sₘ
    inst✝ : IsScalarTower R S Sₘ
    x : R
    y : Subtype fun x => Membership.mem M x
    ⊢ Eq ((algebraMap Rₘ Sₘ) (IsLocalization.mk' Rₘ (HMul.hMul (↑y) x) y)) ((algeb …
  -/
  exact congr_arg (algebraMap Rₘ Sₘ) (IsLocalization.mk'_mul_cancel_left x y)
  /-
    🎉 no goals
  -/


/-- If the square below commutes, the bottom map is uniquely specified:
```
R  →  S
↓     ↓
Rₘ → Sₘ
```
-/
theorem IsLocalization.algebraMap_eq_map_map_submonoid :
    algebraMap Rₘ Sₘ =
      map Sₘ (algebraMap R S)
        (show _ ≤ (Algebra.algebraMapSubmonoid S M).comap _ from M.le_comap_map) :=
  Eq.symm <|
    IsLocalization.map_unique _ (algebraMap Rₘ Sₘ) fun x => by
      /-
        R : Type u_1
        inst✝¹¹ : CommRing R
        M : Submonoid R
        S : Type u_2
        inst✝¹⁰ : CommRing S
        inst✝⁹ : Algebra R S
        Rₘ : Type u_4
        Sₘ : Type u_5
        inst✝⁸ : CommRing Rₘ
        inst✝⁷ : CommRing Sₘ
        inst✝⁶ : Algebra R Rₘ
        inst✝⁵ : IsLocalization M Rₘ
        inst✝⁴ : Algebra S Sₘ
        i : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
        inst✝³ : Algebra Rₘ Sₘ
        inst✝² : Algebra R Sₘ
        inst✝¹ : IsScalarTower R Rₘ Sₘ
        inst✝ : IsScalarTower R S Sₘ
        x : R
        ⊢ Eq ((algebraMap Rₘ Sₘ) ((algebraMap R Rₘ) x)) ((algebraMap S Sₘ) ((algebraMa …
      -/
      rw [← IsScalarTower.algebraMap_apply R S Sₘ, ← IsScalarTower.algebraMap_apply R Rₘ Sₘ]
      /-
        🎉 no goals
      -/


/-- If the square below commutes, the bottom map is uniquely specified:
```
R  →  S
↓     ↓
Rₘ → Sₘ
```
-/
theorem IsLocalization.algebraMap_apply_eq_map_map_submonoid (x) :
    algebraMap Rₘ Sₘ x =
      map Sₘ (algebraMap R S)
        (show _ ≤ (Algebra.algebraMapSubmonoid S M).comap _ from M.le_comap_map) x :=
  DFunLike.congr_fun (IsLocalization.algebraMap_eq_map_map_submonoid _ _ _ _) x


theorem IsLocalization.lift_algebraMap_eq_algebraMap :
    IsLocalization.lift (M := M) (IsLocalization.map_units_map_submonoid S Sₘ) =
      algebraMap Rₘ Sₘ :=
  IsLocalization.lift_unique _ fun _ => (IsScalarTower.algebraMap_apply _ _ _ _).symm


/-- Injectivity of the underlying `algebraMap` descends to the algebra induced by localization. -/
theorem localizationAlgebra_injective (hRS : Function.Injective (algebraMap R S)) :
    Function.Injective (@algebraMap Rₘ Sₘ _ _ (localizationAlgebra M S)) :=
  have : IsLocalization (M.map (algebraMap R S)) Sₘ := i
  IsLocalization.map_injective_of_injective _ _ _ hRS


