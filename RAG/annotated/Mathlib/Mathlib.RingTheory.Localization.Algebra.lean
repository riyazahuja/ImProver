open IsLocalization in
variable (M S) in
/-- The span of `I` in a localization of `R` at `M` is the localization of `I` at `M`. -/
-- TODO: golf using `Ideal.localized'_eq_map`
instance Algebra.idealMap_isLocalizedModule (I : Ideal R) :
    IsLocalizedModule M (Algebra.idealMap I (S := S)) where
  map_units x :=
    (Module.End_isUnit_iff _).mpr ⟨fun a b e ↦ Subtype.ext ((map_units S x).mul_right_injective
          /-
            R : Type u_1
            S : Type u_2
            P : Type u_3
            Q : Type u_4
            inst✝⁷ : CommSemiring R
            inst✝⁶ : CommSemiring S
            inst✝⁵ : CommSemiring P
            inst✝⁴ : CommSemiring Q
            M : Submonoid R
            T : Submonoid P
            inst✝³ : Algebra R S
            inst✝² : Algebra P Q
            inst✝¹ : IsLocalization M S
            inst✝ : IsLocalization T Q
            g : RingHom R P
            I : Ideal R
            x : Subtype fun x => Membership.mem M x
            a b : Subtype fun x => Membership.mem (Ideal.map (algebraMap R S) I) x
            e : Eq (((algebraMap R (Module.End R (Subtype fun x => Membership.mem (Ideal.m …
            ⊢ Eq ((fun x_1 => HMul.hMul ((algebraMap R S) ↑x) x_1) ↑a) ((fun x_1 => HMul.h …
          -/
      (by simpa [Algebra.smul_def] using congr(($e).1))),
          /-
            🎉 no goals
          -/
      fun a ↦ ⟨⟨_, Ideal.mul_mem_left _ (map_units S x).unit⁻¹.1 a.2⟩,
                        /-
                          R : Type u_1
                          S : Type u_2
                          P : Type u_3
                          Q : Type u_4
                          inst✝⁷ : CommSemiring R
                          inst✝⁶ : CommSemiring S
                          inst✝⁵ : CommSemiring P
                          inst✝⁴ : CommSemiring Q
                          M : Submonoid R
                          T : Submonoid P
                          inst✝³ : Algebra R S
                          inst✝² : Algebra P Q
                          inst✝¹ : IsLocalization M S
                          inst✝ : IsLocalization T Q
                          g : RingHom R P
                          I : Ideal R
                          x : Subtype fun x => Membership.mem M x
                          a : Subtype fun x => Membership.mem (Ideal.map (algebraMap R S) I) x
                          ⊢ Eq ↑(((algebraMap R (Module.End R (Subtype fun x => Membership.mem (Ideal.ma …
                        -/
        Subtype.ext (by simp [Algebra.smul_def, ← mul_assoc])⟩⟩
                        /-
                          🎉 no goals
                        -/
  surj' y :=
    have ⟨x, hx⟩ := (mem_map_algebraMap_iff M S).mp y.property
                        /-
                          R : Type u_1
                          S : Type u_2
                          P : Type u_3
                          Q : Type u_4
                          inst✝⁷ : CommSemiring R
                          inst✝⁶ : CommSemiring S
                          inst✝⁵ : CommSemiring P
                          inst✝⁴ : CommSemiring Q
                          M : Submonoid R
                          T : Submonoid P
                          inst✝³ : Algebra R S
                          inst✝² : Algebra P Q
                          inst✝¹ : IsLocalization M S
                          inst✝ : IsLocalization T Q
                          g : RingHom R P
                          I : Ideal R
                          y : Subtype fun x => Membership.mem (Ideal.map (algebraMap R S) I) x
                          x : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.me …
                          hx : Eq (HMul.hMul (↑y) ((algebraMap R S) ↑x.2)) ((algebraMap R S) ↑x.1)
                          ⊢ Eq ↑(HSMul.hSMul x.2 y) ↑((Algebra.idealMap S I) x.1)
                        -/
    ⟨x, Subtype.ext (by simp [Submonoid.smul_def, Algebra.smul_def, mul_comm, hx])⟩
                        /-
                          🎉 no goals
                        -/
  exists_of_eq h := ⟨_, Subtype.ext (exists_of_eq congr(($h).1)).choose_spec⟩


lemma IsLocalization.ker_map (hT : Submonoid.map g M = T) :
    RingHom.ker (IsLocalization.map Q g (hT.symm ▸ M.le_comap_map) : S →+* Q) =
      (RingHom.ker g).map (algebraMap R S) := by
  /-
    R : Type u_1
    S : Type u_2
    P : Type u_3
    Q : Type u_4
    inst✝⁷ : CommSemiring R
    inst✝⁶ : CommSemiring S
    inst✝⁵ : CommSemiring P
    inst✝⁴ : CommSemiring Q
    M : Submonoid R
    T : Submonoid P
    inst✝³ : Algebra R S
    inst✝² : Algebra P Q
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization T Q
    g : RingHom R P
    hT : Eq (Submonoid.map g M) T
    ⊢ Eq (RingHom.ker (IsLocalization.map Q g ⋯)) (Ideal.map (algebraMap R S) (Rin …
  -/
  ext x
  /-
    case h
    R : Type u_1
    S : Type u_2
    P : Type u_3
    Q : Type u_4
    inst✝⁷ : CommSemiring R
    inst✝⁶ : CommSemiring S
    inst✝⁵ : CommSemiring P
    inst✝⁴ : CommSemiring Q
    M : Submonoid R
    T : Submonoid P
    inst✝³ : Algebra R S
    inst✝² : Algebra P Q
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization T Q
    g : RingHom R P
    hT : Eq (Submonoid.map g M) T
    x : S
    ⊢ Iff (Membership.mem (RingHom.ker (IsLocalization.map Q g ⋯)) x) (Membership. …
  -/
  obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective M x
  simp [RingHom.mem_ker, IsLocalization.map_mk', IsLocalization.mk'_eq_zero_iff,
    IsLocalization.mk'_mem_map_algebraMap_iff, ← hT]


variable (S) in
/-- The canonical linear map from the kernel of `g` to the kernel of its localization. -/
def RingHom.toKerIsLocalization (hy : M ≤ Submonoid.comap g T) :
    RingHom.ker g →ₗ[R] RingHom.ker (IsLocalization.map Q g hy : S →+* Q) where
                                   /-
                                     R : Type u_1
                                     S : Type u_2
                                     P : Type u_3
                                     Q : Type u_4
                                     inst✝⁷ : CommSemiring R
                                     inst✝⁶ : CommSemiring S
                                     inst✝⁵ : CommSemiring P
                                     inst✝⁴ : CommSemiring Q
                                     M : Submonoid R
                                     T : Submonoid P
                                     inst✝³ : Algebra R S
                                     inst✝² : Algebra P Q
                                     inst✝¹ : IsLocalization M S
                                     inst✝ : IsLocalization T Q
                                     g : RingHom R P
                                     hy : LE.le M (Submonoid.comap g T)
                                     x : Subtype fun x => Membership.mem (RingHom.ker g) x
                                     ⊢ Membership.mem (RingHom.ker (IsLocalization.map Q g hy)) ((algebraMap R S) ↑x)
                                   -/
  toFun x := ⟨algebraMap R S x, by simp [RingHom.mem_ker, RingHom.mem_ker.mp x.property]⟩
                                   /-
                                     🎉 no goals
                                   -/
  map_add' x y := by
    /-
      R : Type u_1
      S : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝⁷ : CommSemiring R
      inst✝⁶ : CommSemiring S
      inst✝⁵ : CommSemiring P
      inst✝⁴ : CommSemiring Q
      M : Submonoid R
      T : Submonoid P
      inst✝³ : Algebra R S
      inst✝² : Algebra P Q
      inst✝¹ : IsLocalization M S
      inst✝ : IsLocalization T Q
      g : RingHom R P
      hy : LE.le M (Submonoid.comap g T)
      x y : Subtype fun x => Membership.mem (RingHom.ker g) x
      ⊢ Eq ((fun x => ⟨(algebraMap R S) ↑x, ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x  …
    -/
    simp only [Submodule.coe_add, map_add, AddMemClass.mk_add_mk]
    /-
      🎉 no goals
    -/
  map_smul' a x := by
    simp only [SetLike.val_smul, smul_eq_mul, map_mul, id_apply, SetLike.mk_smul_of_tower_mk,
      Algebra.smul_def]


@[simp]
lemma RingHom.toKerIsLocalization_apply (hy : M ≤ Submonoid.comap g T) (r : RingHom.ker g) :
    (RingHom.toKerIsLocalization S Q g hy r).val = algebraMap R S r :=
  rfl


/-- The canonical linear map from the kernel of `g` to the kernel of its localization
is localizing. In other words, localization commutes with taking kernels. -/
lemma RingHom.toKerIsLocalization_isLocalizedModule (hT : Submonoid.map g M = T) :
    IsLocalizedModule M (toKerIsLocalization S Q g (hT.symm ▸ Submonoid.le_comap_map M)) := by
  /-
    R : Type u_1
    S : Type u_2
    P : Type u_3
    Q : Type u_4
    inst✝⁷ : CommSemiring R
    inst✝⁶ : CommSemiring S
    inst✝⁵ : CommSemiring P
    inst✝⁴ : CommSemiring Q
    M : Submonoid R
    T : Submonoid P
    inst✝³ : Algebra R S
    inst✝² : Algebra P Q
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization T Q
    g : RingHom R P
    hT : Eq (Submonoid.map g M) T
    ⊢ IsLocalizedModule M (RingHom.toKerIsLocalization S Q g ⋯)
  -/
  let e := LinearEquiv.ofEq _ _ (IsLocalization.ker_map (S := S) Q g hT).symm
  convert_to IsLocalizedModule M ((e.restrictScalars R).toLinearMap ∘ₗ
    Algebra.idealMap S (RingHom.ker g))
  /-
    R : Type u_1
    S : Type u_2
    P : Type u_3
    Q : Type u_4
    inst✝⁷ : CommSemiring R
    inst✝⁶ : CommSemiring S
    inst✝⁵ : CommSemiring P
    inst✝⁴ : CommSemiring Q
    M : Submonoid R
    T : Submonoid P
    inst✝³ : Algebra R S
    inst✝² : Algebra P Q
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization T Q
    g : RingHom R P
    hT : Eq (Submonoid.map g M) T
    e : LinearEquiv (RingHom.id S) (Subtype fun x => Membership.mem (Ideal.map (al …
    ⊢ IsLocalizedModule M ((↑(LinearEquiv.restrictScalars R e)).comp (Algebra.idea …
  -/
  apply IsLocalizedModule.of_linearEquiv
  /-
    🎉 no goals
  -/


instance isLocalization_algebraMapSubmonoid_map_algHom (f : A →ₐ[R] B) :
    IsLocalization ((algebraMapSubmonoid A M).map f.toRingHom) Bₚ := by
  /-
    R✝ : Type u_1
    S : Type u_2
    P : Type u_3
    Q : Type u_4
    inst✝²⁹ : CommSemiring R✝
    inst✝²⁸ : CommSemiring S
    inst✝²⁷ : CommSemiring P
    inst✝²⁶ : CommSemiring Q
    M✝ : Submonoid R✝
    T : Submonoid P
    inst✝²⁵ : Algebra R✝ S
    inst✝²⁴ : Algebra P Q
    inst✝²³ : IsLocalization M✝ S
    inst✝²² : IsLocalization T Q
    g : RingHom R✝ P
    R : Type u
    inst✝²¹ : CommRing R
    M : Submonoid R
    A : Type v
    inst✝²⁰ : CommRing A
    inst✝¹⁹ : Algebra R A
    B : Type w
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra R B
    Rₚ : Type u'
    inst✝¹⁶ : CommRing Rₚ
    inst✝¹⁵ : Algebra R Rₚ
    inst✝¹⁴ : IsLocalization M Rₚ
    Aₚ : Type v'
    inst✝¹³ : CommRing Aₚ
    inst✝¹² : Algebra R Aₚ
    inst✝¹¹ : Algebra A Aₚ
    inst✝¹⁰ : IsScalarTower R A Aₚ
    inst✝⁹ : IsLocalization (Algebra.algebraMapSubmonoid A M) Aₚ
    Bₚ : Type v'
    inst✝⁸ : CommRing Bₚ
    inst✝⁷ : Algebra R Bₚ
    inst✝⁶ : Algebra B Bₚ
    inst✝⁵ : IsScalarTower R B Bₚ
    inst✝⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₚ
    inst✝³ : Algebra Rₚ Aₚ
    inst✝² : Algebra Rₚ Bₚ
    inst✝¹ : IsScalarTower R Rₚ Aₚ
    inst✝ : IsScalarTower R Rₚ Bₚ
    f : AlgHom R A B
    ⊢ IsLocalization (Submonoid.map f.toRingHom (Algebra.algebraMapSubmonoid A M)) …
  -/
  erw [algebraMapSubmonoid_map_eq M f]
  /-
    R✝ : Type u_1
    S : Type u_2
    P : Type u_3
    Q : Type u_4
    inst✝²⁹ : CommSemiring R✝
    inst✝²⁸ : CommSemiring S
    inst✝²⁷ : CommSemiring P
    inst✝²⁶ : CommSemiring Q
    M✝ : Submonoid R✝
    T : Submonoid P
    inst✝²⁵ : Algebra R✝ S
    inst✝²⁴ : Algebra P Q
    inst✝²³ : IsLocalization M✝ S
    inst✝²² : IsLocalization T Q
    g : RingHom R✝ P
    R : Type u
    inst✝²¹ : CommRing R
    M : Submonoid R
    A : Type v
    inst✝²⁰ : CommRing A
    inst✝¹⁹ : Algebra R A
    B : Type w
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra R B
    Rₚ : Type u'
    inst✝¹⁶ : CommRing Rₚ
    inst✝¹⁵ : Algebra R Rₚ
    inst✝¹⁴ : IsLocalization M Rₚ
    Aₚ : Type v'
    inst✝¹³ : CommRing Aₚ
    inst✝¹² : Algebra R Aₚ
    inst✝¹¹ : Algebra A Aₚ
    inst✝¹⁰ : IsScalarTower R A Aₚ
    inst✝⁹ : IsLocalization (Algebra.algebraMapSubmonoid A M) Aₚ
    Bₚ : Type v'
    inst✝⁸ : CommRing Bₚ
    inst✝⁷ : Algebra R Bₚ
    inst✝⁶ : Algebra B Bₚ
    inst✝⁵ : IsScalarTower R B Bₚ
    inst✝⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₚ
    inst✝³ : Algebra Rₚ Aₚ
    inst✝² : Algebra Rₚ Bₚ
    inst✝¹ : IsScalarTower R Rₚ Aₚ
    inst✝ : IsScalarTower R Rₚ Bₚ
    f : AlgHom R A B
    ⊢ IsLocalization (Algebra.algebraMapSubmonoid B M) Bₚ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- An algebra map `A →ₐ[R] B` induces an algebra map on localizations `Aₚ →ₐ[Rₚ] Bₚ`. -/
noncomputable def mapₐ (f : A →ₐ[R] B) : Aₚ →ₐ[Rₚ] Bₚ :=
  ⟨IsLocalization.map Bₚ f.toRingHom (Algebra.algebraMapSubmonoid_le_comap M f), fun r ↦ by
    /-
      R✝ : Type u_1
      S : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝²⁹ : CommSemiring R✝
      inst✝²⁸ : CommSemiring S
      inst✝²⁷ : CommSemiring P
      inst✝²⁶ : CommSemiring Q
      M✝ : Submonoid R✝
      T : Submonoid P
      inst✝²⁵ : Algebra R✝ S
      inst✝²⁴ : Algebra P Q
      inst✝²³ : IsLocalization M✝ S
      inst✝²² : IsLocalization T Q
      g : RingHom R✝ P
      R : Type u
      inst✝²¹ : CommRing R
      M : Submonoid R
      A : Type v
      inst✝²⁰ : CommRing A
      inst✝¹⁹ : Algebra R A
      B : Type w
      inst✝¹⁸ : CommRing B
      inst✝¹⁷ : Algebra R B
      Rₚ : Type u'
      inst✝¹⁶ : CommRing Rₚ
      inst✝¹⁵ : Algebra R Rₚ
      inst✝¹⁴ : IsLocalization M Rₚ
      Aₚ : Type v'
      inst✝¹³ : CommRing Aₚ
      inst✝¹² : Algebra R Aₚ
      inst✝¹¹ : Algebra A Aₚ
      inst✝¹⁰ : IsScalarTower R A Aₚ
      inst✝⁹ : IsLocalization (Algebra.algebraMapSubmonoid A M) Aₚ
      Bₚ : Type v'
      inst✝⁸ : CommRing Bₚ
      inst✝⁷ : Algebra R Bₚ
      inst✝⁶ : Algebra B Bₚ
      inst✝⁵ : IsScalarTower R B Bₚ
      inst✝⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₚ
      inst✝³ : Algebra Rₚ Aₚ
      inst✝² : Algebra Rₚ Bₚ
      inst✝¹ : IsScalarTower R Rₚ Aₚ
      inst✝ : IsScalarTower R Rₚ Bₚ
      f : AlgHom R A B
      r : Rₚ
      ⊢ Eq ((↑↑(IsLocalization.map Bₚ f.toRingHom ⋯)).toFun ((algebraMap Rₚ Aₚ) r))  …
    -/
    obtain ⟨a, m, rfl⟩ := IsLocalization.mk'_surjective M r
    /-
      case intro.intro
      R✝ : Type u_1
      S : Type u_2
      P : Type u_3
      Q : Type u_4
      inst✝²⁹ : CommSemiring R✝
      inst✝²⁸ : CommSemiring S
      inst✝²⁷ : CommSemiring P
      inst✝²⁶ : CommSemiring Q
      M✝ : Submonoid R✝
      T : Submonoid P
      inst✝²⁵ : Algebra R✝ S
      inst✝²⁴ : Algebra P Q
      inst✝²³ : IsLocalization M✝ S
      inst✝²² : IsLocalization T Q
      g : RingHom R✝ P
      R : Type u
      inst✝²¹ : CommRing R
      M : Submonoid R
      A : Type v
      inst✝²⁰ : CommRing A
      inst✝¹⁹ : Algebra R A
      B : Type w
      inst✝¹⁸ : CommRing B
      inst✝¹⁷ : Algebra R B
      Rₚ : Type u'
      inst✝¹⁶ : CommRing Rₚ
      inst✝¹⁵ : Algebra R Rₚ
      inst✝¹⁴ : IsLocalization M Rₚ
      Aₚ : Type v'
      inst✝¹³ : CommRing Aₚ
      inst✝¹² : Algebra R Aₚ
      inst✝¹¹ : Algebra A Aₚ
      inst✝¹⁰ : IsScalarTower R A Aₚ
      inst✝⁹ : IsLocalization (Algebra.algebraMapSubmonoid A M) Aₚ
      Bₚ : Type v'
      inst✝⁸ : CommRing Bₚ
      inst✝⁷ : Algebra R Bₚ
      inst✝⁶ : Algebra B Bₚ
      inst✝⁵ : IsScalarTower R B Bₚ
      inst✝⁴ : IsLocalization (Algebra.algebraMapSubmonoid B M) Bₚ
      inst✝³ : Algebra Rₚ Aₚ
      inst✝² : Algebra Rₚ Bₚ
      inst✝¹ : IsScalarTower R Rₚ Aₚ
      inst✝ : IsScalarTower R Rₚ Bₚ
      f : AlgHom R A B
      a : R
      m : Subtype fun x => Membership.mem M x
      ⊢ Eq ((↑↑(IsLocalization.map Bₚ f.toRingHom ⋯)).toFun ((algebraMap Rₚ Aₚ) (IsL …
    -/
    simp [algebraMap_mk' (S := A), algebraMap_mk' (S := B), map_mk']⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma mapₐ_coe (f : A →ₐ[R] B) :
    (mapₐ M Rₚ Aₚ Bₚ f : Aₚ → Bₚ) = map Bₚ f.toRingHom (algebraMapSubmonoid_le_comap M f)  :=
  rfl


lemma mapₐ_injective_of_injective (f : A →ₐ[R] B) (hf : Function.Injective f) :
    Function.Injective (mapₐ M Rₚ Aₚ Bₚ f) :=
  IsLocalization.map_injective_of_injective _ _ _ hf


lemma mapₐ_surjective_of_surjective (f : A →ₐ[R] B) (hf : Function.Surjective f) :
    Function.Surjective (mapₐ M Rₚ Aₚ Bₚ f) :=
  IsLocalization.map_surjective_of_surjective _ _ _ hf


/-- The canonical linear map from the kernel of an algebra homomorphism to its localization. -/
def AlgHom.toKerIsLocalization (f : A →ₐ[R] B) :
    RingHom.ker f →ₗ[A] RingHom.ker (mapₐ M Rₚ Aₚ Bₚ f) :=
  RingHom.toKerIsLocalization Aₚ Bₚ f.toRingHom (algebraMapSubmonoid_le_comap M f)


@[simp]
lemma AlgHom.toKerIsLocalization_apply (f : A →ₐ[R] B) (x : RingHom.ker f) :
    AlgHom.toKerIsLocalization M Rₚ Aₚ Bₚ f x =
      RingHom.toKerIsLocalization Aₚ Bₚ f.toRingHom (algebraMapSubmonoid_le_comap M f) x :=
  rfl


/-- The canonical linear map from the kernel of an algebra homomorphism to its localization
is localizing. -/
lemma AlgHom.toKerIsLocalization_isLocalizedModule (f : A →ₐ[R] B) :
    IsLocalizedModule (Algebra.algebraMapSubmonoid A M)
      (AlgHom.toKerIsLocalization M Rₚ Aₚ Bₚ f) :=
  RingHom.toKerIsLocalization_isLocalizedModule Bₚ f.toRingHom
    (algebraMapSubmonoid_map_eq M f)


