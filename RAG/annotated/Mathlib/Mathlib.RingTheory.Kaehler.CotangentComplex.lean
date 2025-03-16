/--
The cotangent space on `P = R[X]`.
This is isomorphic to `Sⁿ` with `n` being the number of variables of `P`.
-/
abbrev CotangentSpace : Type _ := S ⊗[P.Ring] Ω[P.Ring⁄R]


/-- The cotangent complex given by a presentation `R[X] → S` (i.e. a closed embedding `S ↪ Aⁿ`). -/
noncomputable
def cotangentComplex : P.Cotangent →ₗ[S] P.CotangentSpace :=
  letI f : P.Cotangent ≃ₗ[P.Ring] P.ker.Cotangent :=
    { __ := AddEquiv.refl _, map_smul' := Cotangent.val_smul' }
  (kerCotangentToTensor R P.Ring S ∘ₗ f).extendScalarsOfSurjective P.algebraMap_surjective


@[simp]
lemma cotangentComplex_mk (x) : P.cotangentComplex (.mk x) = 1 ⊗ₜ .D _ _ x :=
  kerCotangentToTensor_toCotangent _ _ _ _


/--
This is the map on the cotangent space associated to a map of presentation.
The matrix associated to this map is the Jacobian matrix. See `CotangentSpace.repr_map`.
-/
protected noncomputable
def map (f : Hom P P') : P.CotangentSpace →ₗ[S] P'.CotangentSpace := by
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.CotangentSpace
  -/
  letI := ((algebraMap S S').comp (algebraMap P.Ring S)).toAlgebra
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    this : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).toA …
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.CotangentSpace
  -/
  haveI : IsScalarTower P.Ring S S' := IsScalarTower.of_algebraMap_eq' rfl
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    this✝ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).to …
    this : IsScalarTower P.Ring S S'
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.CotangentSpace
  -/
  letI := f.toAlgHom.toAlgebra
  haveI : IsScalarTower P.Ring P'.Ring S' :=
    IsScalarTower.of_algebraMap_eq (fun x ↦ (f.algebraMap_toRingHom x).symm)
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    this✝² : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
    this✝¹ : IsScalarTower P.Ring S S'
    this✝ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
    this : IsScalarTower P.Ring P'.Ring S'
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.CotangentSpace
  -/
  apply LinearMap.liftBaseChange
  /-
    case l
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    this✝² : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
    this✝¹ : IsScalarTower P.Ring S S'
    this✝ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
    this : IsScalarTower P.Ring P'.Ring S'
    ⊢ LinearMap (RingHom.id P.Ring) (KaehlerDifferential R P.Ring) P'.CotangentSpace
  -/
  refine (TensorProduct.mk _ _ _ 1).restrictScalars _ ∘ₗ KaehlerDifferential.map R R' P.Ring P'.Ring
  /-
    🎉 no goals
  -/


@[simp]
lemma map_tmul (f : Hom P P') (x y) :
    CotangentSpace.map f (x ⊗ₜ .D _ _ y) = (algebraMap _ _ x) ⊗ₜ .D _ _ (f.toAlgHom y) := by
  simp only [CotangentSpace.map, AlgHom.toRingHom_eq_coe, LinearMap.liftBaseChange_tmul,
    LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply, map_D, mk_apply]
  /-
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f : P.Hom P'
    x : S
    y : P.Ring
    ⊢ Eq (HSMul.hSMul x (TensorProduct.tmul P'.Ring 1 ((KaehlerDifferential.D R' P …
  -/
  rw [smul_tmul', ← Algebra.algebraMap_eq_smul_one]
  /-
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f : P.Hom P'
    x : S
    y : P.Ring
    ⊢ Eq (TensorProduct.tmul P'.Ring ((algebraMap S S') x) ((KaehlerDifferential.D …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma map_id :
                                                    /-
                                                      R : Type u
                                                      S : Type v
                                                      inst✝² : CommRing R
                                                      inst✝¹ : CommRing S
                                                      inst✝ : Algebra R S
                                                      P : Algebra.Extension R S
                                                      ⊢ Eq (Algebra.Extension.CotangentSpace.map (Algebra.Extension.Hom.id P)) Linea …
                                                    -/
    CotangentSpace.map (.id P) = LinearMap.id := by ext; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma map_comp (f : Hom P P') (g : Hom P' P'') :
    CotangentSpace.map (g.comp f) =
      (CotangentSpace.map g).restrictScalars S ∘ₗ CotangentSpace.map f := by
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f : P.Hom P'
    g : P'.Hom P''
    ⊢ Eq (Algebra.Extension.CotangentSpace.map (g.comp f)) ((↑S (Algebra.Extension …
  -/
  ext x
  induction x using TensorProduct.induction_on with
  | zero =>
    simp only [map_zero, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply]
  | add =>
    simp only [map_add, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply, *]
  | tmul x y =>
    obtain ⟨y, rfl⟩ := KaehlerDifferential.tensorProductTo_surjective _ _ y
    induction y with
    | zero => simp only [map_zero, tmul_zero, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
        Function.comp_apply]
    | add => simp only [map_add, tmul_add, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
      Function.comp_apply, *]
    | tmul => simp only [Derivation.tensorProductTo_tmul, tmul_smul, smul_tmul', map_tmul,
        Hom.toAlgHom_apply, Hom.comp_toRingHom, RingHom.coe_comp, Function.comp_apply,
        LinearMap.coe_comp, LinearMap.coe_restrictScalars,
        ← IsScalarTower.algebraMap_apply S S' S'']


lemma map_comp_apply (f : Hom P P') (g : Hom P' P'') (x) :
    CotangentSpace.map (g.comp f) x = .map g (.map f x) :=
  DFunLike.congr_fun (map_comp f g) x


lemma map_cotangentComplex (f : Hom P P') (x) :
    CotangentSpace.map f (P.cotangentComplex x) = P'.cotangentComplex (.map f x) := by
  /-
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f : P.Hom P'
    x : P.Cotangent
    ⊢ Eq ((Algebra.Extension.CotangentSpace.map f) (P.cotangentComplex x)) (P'.cot …
  -/
  obtain ⟨x, rfl⟩ := Cotangent.mk_surjective x
  /-
    case intro
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f : P.Hom P'
    x : Subtype fun x => Membership.mem P.ker x
    ⊢ Eq ((Algebra.Extension.CotangentSpace.map f) (P.cotangentComplex (Algebra.Ex …
  -/
  rw [cotangentComplex_mk, map_tmul, map_one, Cotangent.map_mk, cotangentComplex_mk]
  /-
    🎉 no goals
  -/


lemma map_comp_cotangentComplex (f : Hom P P') :
    CotangentSpace.map f ∘ₗ P.cotangentComplex =
      P'.cotangentComplex.restrictScalars S ∘ₗ Cotangent.map f := by
  /-
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f : P.Hom P'
    ⊢ Eq ((Algebra.Extension.CotangentSpace.map f).comp P.cotangentComplex) ((↑S P …
  -/
  ext x; exact map_cotangentComplex f x
         /-
           🎉 no goals
         -/


lemma Hom.sub_aux (f g : Hom P P') (x y) :
    letI := ((algebraMap S S').comp (algebraMap P.Ring S)).toAlgebra
    f.toAlgHom (x * y) - g.toAlgHom (x * y) -
        (P'.σ ((algebraMap P.Ring S') x) * (f.toAlgHom y - g.toAlgHom y) +
          P'.σ ((algebraMap P.Ring S') y) * (f.toAlgHom x - g.toAlgHom x)) ∈
      P'.ker ^ 2 := by
  /-
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f g : P.Hom P'
    x y : P.Ring
    ⊢ Membership.mem (HPow.hPow P'.ker 2) (HSub.hSub (HSub.hSub (f.toAlgHom (HMul. …
  -/
  letI := ((algebraMap S S').comp (algebraMap P.Ring S)).toAlgebra
  have :
      (f.toAlgHom x - P'.σ (algebraMap P.Ring S' x)) * (f.toAlgHom y - g.toAlgHom y) +
      (g.toAlgHom y - P'.σ (algebraMap P.Ring S' y)) * (f.toAlgHom x - g.toAlgHom x)
        ∈ P'.ker ^ 2 := by
    rw [pow_two]
    refine Ideal.add_mem _ (Ideal.mul_mem_mul ?_ ?_) (Ideal.mul_mem_mul ?_ ?_) <;>
      simp only [RingHom.algebraMap_toAlgebra, AlgHom.toRingHom_eq_coe, RingHom.coe_comp,
        RingHom.coe_coe, Function.comp_apply, map_aeval, ← IsScalarTower.algebraMap_eq,
        coe_eval₂Hom, ← aeval_def, ker, RingHom.mem_ker, map_sub, algebraMap_toRingHom,
        algebraMap_σ, sub_self, toAlgHom_apply]
  /-
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f g : P.Hom P'
    x y : P.Ring
    this✝ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).to …
    this : Membership.mem (HPow.hPow P'.ker 2) (HAdd.hAdd (HMul.hMul (HSub.hSub (f …
    ⊢ Membership.mem (HPow.hPow P'.ker 2) (HSub.hSub (HSub.hSub (f.toAlgHom (HMul. …
  -/
  convert this using 1
  /-
    case h.e'_5
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f g : P.Hom P'
    x y : P.Ring
    this✝ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).to …
    this : Membership.mem (HPow.hPow P'.ker 2) (HAdd.hAdd (HMul.hMul (HSub.hSub (f …
    ⊢ Eq (HSub.hSub (HSub.hSub (f.toAlgHom (HMul.hMul x y)) (g.toAlgHom (HMul.hMul …
  -/
  simp only [map_mul]
  /-
    case h.e'_5
    R : Type u
    S : Type v
    inst✝⁹ : CommRing R
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁶ : CommRing R'
    inst✝⁵ : CommRing S'
    inst✝⁴ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : Algebra R S'
    inst✝ : IsScalarTower R R' S'
    f g : P.Hom P'
    x y : P.Ring
    this✝ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).to …
    this : Membership.mem (HPow.hPow P'.ker 2) (HAdd.hAdd (HMul.hMul (HSub.hSub (f …
    ⊢ Eq (HSub.hSub (HSub.hSub (HMul.hMul (f.toAlgHom x) (f.toAlgHom y)) (HMul.hMu …
  -/
  ring
  /-
    🎉 no goals
  -/


/--
If `f` and `g` are two maps `P → P'` between presentations,
then the image of `f - g` is in the kernel of `P' → S`.
-/
@[simps! apply_coe]
noncomputable
def Hom.subToKer (f g : Hom P P') : P.Ring →ₗ[R] P'.ker := by
  refine ((f.toAlgHom.toLinearMap - g.toAlgHom.toLinearMap).codRestrict
    (P'.ker.restrictScalars R) ?_)
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    f g : P.Hom P'
    ⊢ ∀ (c : P.Ring), Membership.mem (Submodule.restrictScalars R P'.ker) ((HSub.h …
  -/
  intro x
  simp only [LinearMap.sub_apply, AlgHom.toLinearMap_apply, ker, algebraMap_eq,
    Submodule.restrictScalars_mem, RingHom.mem_ker, map_sub, RingHom.coe_coe, algebraMap_toRingHom,
    map_aeval, coe_eval₂Hom, sub_self, toAlgHom_apply]


variable [IsScalarTower R S S'] in
/--
If `f` and `g` are two maps `P → P'` between presentations,
their difference induces a map `P.CotangentSpace →ₗ[S] P'.Cotangent` that makes two maps
between the cotangent complexes homotopic.
-/
noncomputable
def Hom.sub (f g : Hom P P') : P.CotangentSpace →ₗ[S] P'.Cotangent := by
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.Cotangent
  -/
  letI := ((algebraMap S S').comp (algebraMap P.Ring S)).toAlgebra
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    this : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).toA …
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.Cotangent
  -/
  haveI : IsScalarTower P.Ring S S' := IsScalarTower.of_algebraMap_eq' rfl
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    this✝ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).to …
    this : IsScalarTower P.Ring S S'
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.Cotangent
  -/
  letI := f.toAlgHom.toAlgebra
  haveI : IsScalarTower P.Ring P'.Ring S' :=
    IsScalarTower.of_algebraMap_eq fun x ↦ (f.algebraMap_toRingHom x).symm
  haveI : IsScalarTower R P.Ring S' :=
    IsScalarTower.of_algebraMap_eq fun x ↦
      show algebraMap R S' x = algebraMap S S' (algebraMap P.Ring S (algebraMap R P.Ring x)) by
        rw [← IsScalarTower.algebraMap_apply R P.Ring S, ← IsScalarTower.algebraMap_apply]
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
    this✝² : IsScalarTower P.Ring S S'
    this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
    this✝ : IsScalarTower P.Ring P'.Ring S'
    this : IsScalarTower R P.Ring S'
    ⊢ LinearMap (RingHom.id S) P.CotangentSpace P'.Cotangent
  -/
  refine (Derivation.liftKaehlerDifferential ?_).liftBaseChange S
  refine
  { __ := Cotangent.mk.restrictScalars R ∘ₗ f.subToKer g
    map_one_eq_zero' := ?_
    leibniz' := ?_ }
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝²³ : CommRing R
      inst✝²² : CommRing S
      inst✝²¹ : Algebra R S
      P : Algebra.Extension R S
      R' : Type u'
      S' : Type v'
      inst✝²⁰ : CommRing R'
      inst✝¹⁹ : CommRing S'
      inst✝¹⁸ : Algebra R' S'
      P' : Algebra.Extension R' S'
      inst✝¹⁷ : Algebra R R'
      inst✝¹⁶ : Algebra S S'
      inst✝¹⁵ : Algebra R S'
      inst✝¹⁴ : IsScalarTower R R' S'
      R'' : Type u''
      S'' : Type v''
      inst✝¹³ : CommRing R''
      inst✝¹² : CommRing S''
      inst✝¹¹ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝¹⁰ : Algebra R R''
      inst✝⁹ : Algebra S S''
      inst✝⁸ : Algebra R S''
      inst✝⁷ : IsScalarTower R R'' S''
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra S' S''
      inst✝⁴ : Algebra R' S''
      inst✝³ : IsScalarTower R' R'' S''
      inst✝² : IsScalarTower R R' R''
      inst✝¹ : IsScalarTower S S' S''
      inst✝ : IsScalarTower R S S'
      f g : P.Hom P'
      this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
      this✝² : IsScalarTower P.Ring S S'
      this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
      this✝ : IsScalarTower P.Ring P'.Ring S'
      this : IsScalarTower R P.Ring S'
      ⊢ Eq (__spread✝⁻⁰ 1) 0
    -/
  · ext
    simp only [LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply,
      Cotangent.val_mk, Cotangent.val_zero, Ideal.toCotangent_eq_zero]
    /-
      case refine_1.e
      R : Type u
      S : Type v
      inst✝²³ : CommRing R
      inst✝²² : CommRing S
      inst✝²¹ : Algebra R S
      P : Algebra.Extension R S
      R' : Type u'
      S' : Type v'
      inst✝²⁰ : CommRing R'
      inst✝¹⁹ : CommRing S'
      inst✝¹⁸ : Algebra R' S'
      P' : Algebra.Extension R' S'
      inst✝¹⁷ : Algebra R R'
      inst✝¹⁶ : Algebra S S'
      inst✝¹⁵ : Algebra R S'
      inst✝¹⁴ : IsScalarTower R R' S'
      R'' : Type u''
      S'' : Type v''
      inst✝¹³ : CommRing R''
      inst✝¹² : CommRing S''
      inst✝¹¹ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝¹⁰ : Algebra R R''
      inst✝⁹ : Algebra S S''
      inst✝⁸ : Algebra R S''
      inst✝⁷ : IsScalarTower R R'' S''
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra S' S''
      inst✝⁴ : Algebra R' S''
      inst✝³ : IsScalarTower R' R'' S''
      inst✝² : IsScalarTower R R' R''
      inst✝¹ : IsScalarTower S S' S''
      inst✝ : IsScalarTower R S S'
      f g : P.Hom P'
      this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
      this✝² : IsScalarTower P.Ring S S'
      this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
      this✝ : IsScalarTower P.Ring P'.Ring S'
      this : IsScalarTower R P.Ring S'
      ⊢ Membership.mem (HPow.hPow P'.ker 2) ↑((f.subToKer g) 1)
    -/
    erw [LinearMap.codRestrict_apply]
    /-
      case refine_1.e
      R : Type u
      S : Type v
      inst✝²³ : CommRing R
      inst✝²² : CommRing S
      inst✝²¹ : Algebra R S
      P : Algebra.Extension R S
      R' : Type u'
      S' : Type v'
      inst✝²⁰ : CommRing R'
      inst✝¹⁹ : CommRing S'
      inst✝¹⁸ : Algebra R' S'
      P' : Algebra.Extension R' S'
      inst✝¹⁷ : Algebra R R'
      inst✝¹⁶ : Algebra S S'
      inst✝¹⁵ : Algebra R S'
      inst✝¹⁴ : IsScalarTower R R' S'
      R'' : Type u''
      S'' : Type v''
      inst✝¹³ : CommRing R''
      inst✝¹² : CommRing S''
      inst✝¹¹ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝¹⁰ : Algebra R R''
      inst✝⁹ : Algebra S S''
      inst✝⁸ : Algebra R S''
      inst✝⁷ : IsScalarTower R R'' S''
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra S' S''
      inst✝⁴ : Algebra R' S''
      inst✝³ : IsScalarTower R' R'' S''
      inst✝² : IsScalarTower R R' R''
      inst✝¹ : IsScalarTower S S' S''
      inst✝ : IsScalarTower R S S'
      f g : P.Hom P'
      this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
      this✝² : IsScalarTower P.Ring S S'
      this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
      this✝ : IsScalarTower P.Ring P'.Ring S'
      this : IsScalarTower R P.Ring S'
      ⊢ Membership.mem (HPow.hPow P'.ker 2) ((HSub.hSub f.toAlgHom.toLinearMap g.toA …
    -/
    simp only [LinearMap.sub_apply, AlgHom.toLinearMap_apply, map_one, sub_self, Submodule.zero_mem]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝²³ : CommRing R
      inst✝²² : CommRing S
      inst✝²¹ : Algebra R S
      P : Algebra.Extension R S
      R' : Type u'
      S' : Type v'
      inst✝²⁰ : CommRing R'
      inst✝¹⁹ : CommRing S'
      inst✝¹⁸ : Algebra R' S'
      P' : Algebra.Extension R' S'
      inst✝¹⁷ : Algebra R R'
      inst✝¹⁶ : Algebra S S'
      inst✝¹⁵ : Algebra R S'
      inst✝¹⁴ : IsScalarTower R R' S'
      R'' : Type u''
      S'' : Type v''
      inst✝¹³ : CommRing R''
      inst✝¹² : CommRing S''
      inst✝¹¹ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝¹⁰ : Algebra R R''
      inst✝⁹ : Algebra S S''
      inst✝⁸ : Algebra R S''
      inst✝⁷ : IsScalarTower R R'' S''
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra S' S''
      inst✝⁴ : Algebra R' S''
      inst✝³ : IsScalarTower R' R'' S''
      inst✝² : IsScalarTower R R' R''
      inst✝¹ : IsScalarTower S S' S''
      inst✝ : IsScalarTower R S S'
      f g : P.Hom P'
      this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
      this✝² : IsScalarTower P.Ring S S'
      this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
      this✝ : IsScalarTower P.Ring P'.Ring S'
      this : IsScalarTower R P.Ring S'
      ⊢ ∀ (a b : P.Ring), Eq (__spread✝⁻⁰ (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a …
    -/
  · intro x y
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝²³ : CommRing R
      inst✝²² : CommRing S
      inst✝²¹ : Algebra R S
      P : Algebra.Extension R S
      R' : Type u'
      S' : Type v'
      inst✝²⁰ : CommRing R'
      inst✝¹⁹ : CommRing S'
      inst✝¹⁸ : Algebra R' S'
      P' : Algebra.Extension R' S'
      inst✝¹⁷ : Algebra R R'
      inst✝¹⁶ : Algebra S S'
      inst✝¹⁵ : Algebra R S'
      inst✝¹⁴ : IsScalarTower R R' S'
      R'' : Type u''
      S'' : Type v''
      inst✝¹³ : CommRing R''
      inst✝¹² : CommRing S''
      inst✝¹¹ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝¹⁰ : Algebra R R''
      inst✝⁹ : Algebra S S''
      inst✝⁸ : Algebra R S''
      inst✝⁷ : IsScalarTower R R'' S''
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra S' S''
      inst✝⁴ : Algebra R' S''
      inst✝³ : IsScalarTower R' R'' S''
      inst✝² : IsScalarTower R R' R''
      inst✝¹ : IsScalarTower S S' S''
      inst✝ : IsScalarTower R S S'
      f g : P.Hom P'
      this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
      this✝² : IsScalarTower P.Ring S S'
      this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
      this✝ : IsScalarTower P.Ring P'.Ring S'
      this : IsScalarTower R P.Ring S'
      x y : P.Ring
      ⊢ Eq (__spread✝⁻⁰ (HMul.hMul x y)) (HAdd.hAdd (HSMul.hSMul x (__spread✝⁻⁰ y))  …
    -/
    ext
    simp only [LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply,
      Cotangent.val_mk, Cotangent.val_add, Cotangent.val_smul''', ← map_smul, ← map_add,
      Ideal.toCotangent_eq, AddSubmonoid.coe_add, Submodule.coe_toAddSubmonoid,
      SetLike.val_smul, smul_eq_mul]
    /-
      case refine_2.e
      R : Type u
      S : Type v
      inst✝²³ : CommRing R
      inst✝²² : CommRing S
      inst✝²¹ : Algebra R S
      P : Algebra.Extension R S
      R' : Type u'
      S' : Type v'
      inst✝²⁰ : CommRing R'
      inst✝¹⁹ : CommRing S'
      inst✝¹⁸ : Algebra R' S'
      P' : Algebra.Extension R' S'
      inst✝¹⁷ : Algebra R R'
      inst✝¹⁶ : Algebra S S'
      inst✝¹⁵ : Algebra R S'
      inst✝¹⁴ : IsScalarTower R R' S'
      R'' : Type u''
      S'' : Type v''
      inst✝¹³ : CommRing R''
      inst✝¹² : CommRing S''
      inst✝¹¹ : Algebra R'' S''
      P'' : Algebra.Extension R'' S''
      inst✝¹⁰ : Algebra R R''
      inst✝⁹ : Algebra S S''
      inst✝⁸ : Algebra R S''
      inst✝⁷ : IsScalarTower R R'' S''
      inst✝⁶ : Algebra R' R''
      inst✝⁵ : Algebra S' S''
      inst✝⁴ : Algebra R' S''
      inst✝³ : IsScalarTower R' R'' S''
      inst✝² : IsScalarTower R R' R''
      inst✝¹ : IsScalarTower S S' S''
      inst✝ : IsScalarTower R S S'
      f g : P.Hom P'
      this✝³ : Algebra P.Ring S' := ((algebraMap S S').comp (algebraMap P.Ring S)).t …
      this✝² : IsScalarTower P.Ring S S'
      this✝¹ : Algebra P.Ring P'.Ring := f.toAlgHom.toAlgebra
      this✝ : IsScalarTower P.Ring P'.Ring S'
      this : IsScalarTower R P.Ring S'
      x y : P.Ring
      ⊢ Membership.mem (HPow.hPow P'.ker 2) (HSub.hSub ↑((f.subToKer g) (HMul.hMul x …
    -/
    exact Hom.sub_aux f g x y
    /-
      🎉 no goals
    -/


lemma Hom.sub_one_tmul (f g : Hom P P') (x) :
    f.sub g (1 ⊗ₜ .D _ _ x) = Cotangent.mk (f.subToKer g x) := by
  simp only [sub, LinearMap.liftBaseChange_tmul, Derivation.liftKaehlerDifferential_comp_D,
    Derivation.mk_coe, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply,
    one_smul]


@[simp]
lemma Hom.sub_tmul (f g : Hom P P') (r x) :
    f.sub g (r ⊗ₜ .D _ _ x) = r • Cotangent.mk (f.subToKer g x) := by
  simp only [sub, LinearMap.liftBaseChange_tmul, Derivation.liftKaehlerDifferential_comp_D,
    Derivation.mk_coe, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply]


lemma CotangentSpace.map_sub_map (f g : Hom P P') :
    CotangentSpace.map f - CotangentSpace.map g =
      P'.cotangentComplex.restrictScalars S ∘ₗ (f.sub g) := by
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    ⊢ Eq (HSub.hSub (Algebra.Extension.CotangentSpace.map f) (Algebra.Extension.Co …
  -/
  ext x
  induction x using TensorProduct.induction_on with
  | zero =>
    simp only [map_zero, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply]
  | add =>
    simp only [map_add, LinearMap.coe_comp, LinearMap.coe_restrictScalars, Function.comp_apply, *]
  | tmul x y =>
    obtain ⟨y, rfl⟩ := KaehlerDifferential.tensorProductTo_surjective _ _ y
    induction y with
    | zero => simp only [map_zero, tmul_zero, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
        Function.comp_apply]
    | add => simp only [map_add, tmul_add, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
      Function.comp_apply, *]
    | tmul =>
      simp only [Derivation.tensorProductTo_tmul, tmul_smul, smul_tmul', LinearMap.sub_apply,
        map_tmul, Hom.toAlgHom_apply, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
        Function.comp_apply, Hom.sub_tmul, LinearMap.map_smul_of_tower, cotangentComplex_mk,
        Hom.subToKer_apply_coe, map_sub, ← algebraMap_eq_smul_one, tmul_sub, smul_sub]


lemma Cotangent.map_sub_map (f g : Hom P P') :
    map f - map g = (f.sub g) ∘ₗ P.cotangentComplex := by
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    ⊢ Eq (HSub.hSub (Algebra.Extension.Cotangent.map f) (Algebra.Extension.Cotange …
  -/
  ext x
  /-
    case h.e
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    x : P.Cotangent
    ⊢ Eq ((HSub.hSub (Algebra.Extension.Cotangent.map f) (Algebra.Extension.Cotang …
  -/
  obtain ⟨x, rfl⟩ := mk_surjective x
  simp only [LinearMap.sub_apply, map_mk, LinearMap.coe_comp, Function.comp_apply,
    cotangentComplex_mk, Hom.sub_tmul, one_smul, val_mk]
  /-
    case h.e.intro
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    x : Subtype fun x => Membership.mem P.ker x
    ⊢ Eq (HSub.hSub (Algebra.Extension.Cotangent.mk ⟨f.toAlgHom ↑x, ⋯⟩) (Algebra.E …
  -/
  apply (Ideal.cotangentEquivIdeal _).injective
  /-
    case h.e.intro.a
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    x : Subtype fun x => Membership.mem P.ker x
    ⊢ Eq (P'.ker.cotangentEquivIdeal (HSub.hSub (Algebra.Extension.Cotangent.mk ⟨f …
  -/
  ext
  simp only [val_sub, val_mk, map_sub, AddSubgroupClass.coe_sub, Ideal.cotangentEquivIdeal_apply,
    Ideal.toCotangent_to_quotient_square, Submodule.mkQ_apply, Ideal.Quotient.mk_eq_mk,
    Hom.subToKer_apply_coe]
  /-
    case h.e.intro.a.a
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f g : P.Hom P'
    x : Subtype fun x => Membership.mem P.ker x
    ⊢ Eq (HSub.hSub ((Ideal.Quotient.mk (HPow.hPow P'.ker 2)) (f.toAlgHom ↑x)) ((I …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable (P) in
/-- The projection map from the relative cotangent space to the module of differentials. -/
noncomputable
abbrev toKaehler : P.CotangentSpace →ₗ[S] Ω[S⁄R] := mapBaseChange _ _ _


lemma toKaehler_surjective : Function.Surjective P.toKaehler :=
  mapBaseChange_surjective _ _ _ P.algebraMap_surjective


lemma exact_cotangentComplex_toKaehler : Function.Exact P.cotangentComplex P.toKaehler :=
  exact_kerCotangentToTensor_mapBaseChange _ _ _ P.algebraMap_surjective


variable (P) in
/--
The first homology of the (naive) cotangent complex of `S` over `R`,
induced by a given presentation `0 → I → P → R → 0`,
defined as the kernel of `I/I² → S ⊗[P] Ω[P⁄R]`.
-/
protected noncomputable
def H1Cotangent : Type _ := LinearMap.ker P.cotangentComplex


noncomputable
                                            /-
                                              R : Type u
                                              S : Type v
                                              inst✝²³ : CommRing R
                                              inst✝²² : CommRing S
                                              inst✝²¹ : Algebra R S
                                              P✝ : Algebra.Extension R S
                                              R' : Type u'
                                              S' : Type v'
                                              inst✝²⁰ : CommRing R'
                                              inst✝¹⁹ : CommRing S'
                                              inst✝¹⁸ : Algebra R' S'
                                              P' : Algebra.Extension R' S'
                                              inst✝¹⁷ : Algebra R R'
                                              inst✝¹⁶ : Algebra S S'
                                              inst✝¹⁵ : Algebra R S'
                                              inst✝¹⁴ : IsScalarTower R R' S'
                                              R'' : Type u''
                                              S'' : Type v''
                                              inst✝¹³ : CommRing R''
                                              inst✝¹² : CommRing S''
                                              inst✝¹¹ : Algebra R'' S''
                                              P'' : Algebra.Extension R'' S''
                                              inst✝¹⁰ : Algebra R R''
                                              inst✝⁹ : Algebra S S''
                                              inst✝⁸ : Algebra R S''
                                              inst✝⁷ : IsScalarTower R R'' S''
                                              inst✝⁶ : Algebra R' R''
                                              inst✝⁵ : Algebra S' S''
                                              inst✝⁴ : Algebra R' S''
                                              inst✝³ : IsScalarTower R' R'' S''
                                              inst✝² : IsScalarTower R R' R''
                                              inst✝¹ : IsScalarTower S S' S''
                                              inst✝ : IsScalarTower R S S'
                                              P : Algebra.Extension R S
                                              ⊢ AddCommGroup P.H1Cotangent
                                            -/
instance : AddCommGroup P.H1Cotangent := by delta Extension.H1Cotangent; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


noncomputable
instance {R₀} [CommRing R₀] [Algebra R₀ S] [Module R₀ P.Cotangent]
    [IsScalarTower R₀ S P.Cotangent] : Module R₀ P.H1Cotangent := by
  /-
    R : Type u
    S : Type v
    inst✝²⁷ : CommRing R
    inst✝²⁶ : CommRing S
    inst✝²⁵ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁴ : CommRing R'
    inst✝²³ : CommRing S'
    inst✝²² : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝²¹ : Algebra R R'
    inst✝²⁰ : Algebra S S'
    inst✝¹⁹ : Algebra R S'
    inst✝¹⁸ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹⁷ : CommRing R''
    inst✝¹⁶ : CommRing S''
    inst✝¹⁵ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁴ : Algebra R R''
    inst✝¹³ : Algebra S S''
    inst✝¹² : Algebra R S''
    inst✝¹¹ : IsScalarTower R R'' S''
    inst✝¹⁰ : Algebra R' R''
    inst✝⁹ : Algebra S' S''
    inst✝⁸ : Algebra R' S''
    inst✝⁷ : IsScalarTower R' R'' S''
    inst✝⁶ : IsScalarTower R R' R''
    inst✝⁵ : IsScalarTower S S' S''
    inst✝⁴ : IsScalarTower R S S'
    P : Algebra.Extension R S
    R₀ : Type ?u.417322
    inst✝³ : CommRing R₀
    inst✝² : Algebra R₀ S
    inst✝¹ : Module R₀ P.Cotangent
    inst✝ : IsScalarTower R₀ S P.Cotangent
    ⊢ Module R₀ P.H1Cotangent
  -/
  delta Extension.H1Cotangent; infer_instance
                               /-
                                 🎉 no goals
                               -/


@[simp] lemma H1Cotangent.val_add (x y : P.H1Cotangent) : (x + y).1 = x.1 + y.1 := rfl

@[simp] lemma H1Cotangent.val_zero : (0 : P.H1Cotangent).1 = 0 := rfl

@[simp] lemma H1Cotangent.val_smul {R₀} [CommRing R₀] [Algebra R₀ S] [Module R₀ P.Cotangent]
    [IsScalarTower R₀ S P.Cotangent] (r : R₀) (x : P.H1Cotangent) : (r • x).1 = r • x.1 := rfl


noncomputable
instance {R₁ R₂} [CommRing R₁] [CommRing R₂] [Algebra R₁ R₂]
    [Algebra R₁ S] [Algebra R₂ S]
    [Module R₁ P.Cotangent] [IsScalarTower R₁ S P.Cotangent]
    [Module R₂ P.Cotangent] [IsScalarTower R₂ S P.Cotangent]
    [IsScalarTower R₁ R₂ P.Cotangent] :
    IsScalarTower R₁ R₂ P.H1Cotangent := by
  /-
    R : Type u
    S : Type v
    inst✝³³ : CommRing R
    inst✝³² : CommRing S
    inst✝³¹ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝³⁰ : CommRing R'
    inst✝²⁹ : CommRing S'
    inst✝²⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝²⁷ : Algebra R R'
    inst✝²⁶ : Algebra S S'
    inst✝²⁵ : Algebra R S'
    inst✝²⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝²³ : CommRing R''
    inst✝²² : CommRing S''
    inst✝²¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝²⁰ : Algebra R R''
    inst✝¹⁹ : Algebra S S''
    inst✝¹⁸ : Algebra R S''
    inst✝¹⁷ : IsScalarTower R R'' S''
    inst✝¹⁶ : Algebra R' R''
    inst✝¹⁵ : Algebra S' S''
    inst✝¹⁴ : Algebra R' S''
    inst✝¹³ : IsScalarTower R' R'' S''
    inst✝¹² : IsScalarTower R R' R''
    inst✝¹¹ : IsScalarTower S S' S''
    inst✝¹⁰ : IsScalarTower R S S'
    P : Algebra.Extension R S
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁹ : CommRing R₁
    inst✝⁸ : CommRing R₂
    inst✝⁷ : Algebra R₁ R₂
    inst✝⁶ : Algebra R₁ S
    inst✝⁵ : Algebra R₂ S
    inst✝⁴ : Module R₁ P.Cotangent
    inst✝³ : IsScalarTower R₁ S P.Cotangent
    inst✝² : Module R₂ P.Cotangent
    inst✝¹ : IsScalarTower R₂ S P.Cotangent
    inst✝ : IsScalarTower R₁ R₂ P.Cotangent
    ⊢ IsScalarTower R₁ R₂ P.H1Cotangent
  -/
  delta Extension.H1Cotangent; infer_instance
                               /-
                                 🎉 no goals
                               -/


lemma subsingleton_h1Cotangent (P : Extension R S) :
    Subsingleton P.H1Cotangent ↔ Function.Injective P.cotangentComplex := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    ⊢ Iff (Subsingleton P.H1Cotangent) (Function.Injective ⇑P.cotangentComplex)
  -/
  delta Extension.H1Cotangent
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    ⊢ Iff (Subsingleton (Subtype fun x => Membership.mem (LinearMap.ker P.cotangen …
  -/
  rw [← LinearMap.ker_eq_bot, Submodule.eq_bot_iff, subsingleton_iff_forall_eq 0, Subtype.forall']
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Extension R S
    ⊢ Iff (∀ (y : Subtype fun x => Membership.mem (LinearMap.ker P.cotangentComple …
  -/
  simp only [Subtype.ext_iff, Submodule.coe_zero]
  /-
    🎉 no goals
  -/


/-- The inclusion of `H¹(L_{S/R})` into the conormal space of a presentation. -/
@[simps!] def h1Cotangentι : P.H1Cotangent →ₗ[S] P.Cotangent := Submodule.subtype _


lemma h1Cotangentι_injective : Function.Injective P.h1Cotangentι := Subtype.val_injective


@[ext] lemma h1Cotangentι_ext (x y : P.H1Cotangent) (e : x.1 = y.1) : x = y := Subtype.ext e


/--
The induced map on the first homology of the (naive) cotangent complex.
-/
@[simps!]
noncomputable
def H1Cotangent.map (f : Hom P P') : P.H1Cotangent →ₗ[S] P'.H1Cotangent := by
  refine (Cotangent.map f).restrict (p := LinearMap.ker P.cotangentComplex)
    (q := (LinearMap.ker P'.cotangentComplex).restrictScalars S) fun x hx ↦ ?_
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f : P.Hom P'
    x : P.Cotangent
    hx : Membership.mem (LinearMap.ker P.cotangentComplex) x
    ⊢ Membership.mem (Submodule.restrictScalars S (LinearMap.ker P'.cotangentCompl …
  -/
  simp only [LinearMap.mem_ker, Submodule.restrictScalars_mem] at hx ⊢
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f : P.Hom P'
    x : P.Cotangent
    hx : Eq (P.cotangentComplex x) 0
    ⊢ Eq (P'.cotangentComplex ((Algebra.Extension.Cotangent.map f) x)) 0
  -/
  apply_fun (CotangentSpace.map f) at hx
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f : P.Hom P'
    x : P.Cotangent
    hx : Eq ((Algebra.Extension.CotangentSpace.map f) (P.cotangentComplex x)) ((Al …
    ⊢ Eq (P'.cotangentComplex ((Algebra.Extension.Cotangent.map f) x)) 0
  -/
  rw [CotangentSpace.map_cotangentComplex] at hx
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f : P.Hom P'
    x : P.Cotangent
    hx : Eq (P'.cotangentComplex ((Algebra.Extension.Cotangent.map f) x)) ((Algebr …
    ⊢ Eq (P'.cotangentComplex ((Algebra.Extension.Cotangent.map f) x)) 0
  -/
  rw [hx]
  /-
    R : Type u
    S : Type v
    inst✝²³ : CommRing R
    inst✝²² : CommRing S
    inst✝²¹ : Algebra R S
    P✝ : Algebra.Extension R S
    R' : Type u'
    S' : Type v'
    inst✝²⁰ : CommRing R'
    inst✝¹⁹ : CommRing S'
    inst✝¹⁸ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁷ : Algebra R R'
    inst✝¹⁶ : Algebra S S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹³ : CommRing R''
    inst✝¹² : CommRing S''
    inst✝¹¹ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝¹⁰ : Algebra R R''
    inst✝⁹ : Algebra S S''
    inst✝⁸ : Algebra R S''
    inst✝⁷ : IsScalarTower R R'' S''
    inst✝⁶ : Algebra R' R''
    inst✝⁵ : Algebra S' S''
    inst✝⁴ : Algebra R' S''
    inst✝³ : IsScalarTower R' R'' S''
    inst✝² : IsScalarTower R R' R''
    inst✝¹ : IsScalarTower S S' S''
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f : P.Hom P'
    x : P.Cotangent
    hx : Eq (P'.cotangentComplex ((Algebra.Extension.Cotangent.map f) x)) ((Algebr …
    ⊢ Eq ((Algebra.Extension.CotangentSpace.map f) 0) 0
  -/
  exact LinearMap.map_zero _
  /-
    🎉 no goals
  -/


lemma H1Cotangent.map_eq (f g : Hom P P') : map f = map g := by
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f g : P.Hom P'
    ⊢ Eq (Algebra.Extension.H1Cotangent.map f) (Algebra.Extension.H1Cotangent.map g)
  -/
  ext x
  /-
    case h.e.e
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f g : P.Hom P'
    x : P.H1Cotangent
    ⊢ Eq (↑((Algebra.Extension.H1Cotangent.map f) x)).val (↑((Algebra.Extension.H1 …
  -/
  simp only [map_apply_coe]
  /-
    case h.e.e
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    P : Algebra.Extension R S
    f g : P.Hom P'
    x : P.H1Cotangent
    ⊢ Eq ((Algebra.Extension.Cotangent.map f) ↑x).val ((Algebra.Extension.Cotangen …
  -/
  rw [← sub_eq_zero, ← Cotangent.val_sub, ← LinearMap.sub_apply, Cotangent.map_sub_map]
  simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.map_coe_ker, map_zero,
    Cotangent.val_zero]


                                                                    /-
                                                                      R : Type u
                                                                      S : Type v
                                                                      inst✝² : CommRing R
                                                                      inst✝¹ : CommRing S
                                                                      inst✝ : Algebra R S
                                                                      P : Algebra.Extension R S
                                                                      ⊢ Eq (Algebra.Extension.H1Cotangent.map (Algebra.Extension.Hom.id P)) LinearMa …
                                                                    -/
@[simp] lemma H1Cotangent.map_id : map (.id P) = LinearMap.id := by ext; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


omit [IsScalarTower R S S'] in
lemma H1Cotangent.map_comp
    (f : Hom P P') (g : Hom P' P'') :
    map (g.comp f) = (map g).restrictScalars S ∘ₗ map f := by
  /-
    R : Type u
    S : Type v
    inst✝²² : CommRing R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    R' : Type u'
    S' : Type v'
    inst✝¹⁹ : CommRing R'
    inst✝¹⁸ : CommRing S'
    inst✝¹⁷ : Algebra R' S'
    P' : Algebra.Extension R' S'
    inst✝¹⁶ : Algebra R R'
    inst✝¹⁵ : Algebra S S'
    inst✝¹⁴ : Algebra R S'
    inst✝¹³ : IsScalarTower R R' S'
    R'' : Type u''
    S'' : Type v''
    inst✝¹² : CommRing R''
    inst✝¹¹ : CommRing S''
    inst✝¹⁰ : Algebra R'' S''
    P'' : Algebra.Extension R'' S''
    inst✝⁹ : Algebra R R''
    inst✝⁸ : Algebra S S''
    inst✝⁷ : Algebra R S''
    inst✝⁶ : IsScalarTower R R'' S''
    inst✝⁵ : Algebra R' R''
    inst✝⁴ : Algebra S' S''
    inst✝³ : Algebra R' S''
    inst✝² : IsScalarTower R' R'' S''
    inst✝¹ : IsScalarTower R R' R''
    inst✝ : IsScalarTower S S' S''
    P : Algebra.Extension R S
    f : P.Hom P'
    g : P'.Hom P''
    ⊢ Eq (Algebra.Extension.H1Cotangent.map (g.comp f)) ((↑S (Algebra.Extension.H1 …
  -/
  ext; simp [Cotangent.map_comp]
       /-
         🎉 no goals
       -/


/-- The canonical basis on the `CotangentSpace`. -/
noncomputable
def cotangentSpaceBasis : Basis P.vars S P.toExtension.CotangentSpace :=
  (mvPolynomialBasis _ _).baseChange _


@[simp]
lemma cotangentSpaceBasis_repr_tmul (r x i) :
    P.cotangentSpaceBasis.repr (r ⊗ₜ[P.Ring] KaehlerDifferential.D R P.Ring x : _) i =
      r * aeval P.val (pderiv i x) := by
  classical
  simp only [cotangentSpaceBasis, Basis.baseChange_repr_tmul, mvPolynomialBasis_repr_apply,
    Algebra.smul_def, mul_comm r, algebraMap_apply, toExtension]


lemma cotangentSpaceBasis_repr_one_tmul (x i) :
    P.cotangentSpaceBasis.repr (1 ⊗ₜ .D _ _ x) i = aeval P.val (pderiv i x) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    x : P.toExtension.Ring
    i : P.vars
    ⊢ Eq ((P.cotangentSpaceBasis.repr (TensorProduct.tmul P.toExtension.Ring 1 ((K …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma cotangentSpaceBasis_apply (i) :
    P.cotangentSpaceBasis i = ((1 : S) ⊗ₜ[P.Ring] D R P.Ring (.X i) : _) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    i : P.vars
    ⊢ Eq (P.cotangentSpaceBasis i) (TensorProduct.tmul P.Ring 1 ((KaehlerDifferent …
  -/
  simp [cotangentSpaceBasis, toExtension]
  /-
    🎉 no goals
  -/


@[simp]
lemma repr_CotangentSpaceMap (f : Hom P P') (i j) :
    P'.cotangentSpaceBasis.repr (CotangentSpace.map f.toExtensionHom (P.cotangentSpaceBasis i)) j =
      aeval P'.val (pderiv j (f.val i)) := by
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    i : P.vars
    j : P'.vars
    ⊢ Eq ((P'.cotangentSpaceBasis.repr ((Algebra.Extension.CotangentSpace.map f.to …
  -/
  rw [cotangentSpaceBasis_apply]
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    i : P.vars
    j : P'.vars
    ⊢ Eq ((P'.cotangentSpaceBasis.repr ((Algebra.Extension.CotangentSpace.map f.to …
  -/
  simp only [toExtension]
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    i : P.vars
    j : P'.vars
    ⊢ Eq ((P'.cotangentSpaceBasis.repr ((Algebra.Extension.CotangentSpace.map f.to …
  -/
  rw [CotangentSpace.map_tmul, map_one]
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    P : Algebra.Generators R S
    R' : Type u'
    S' : Type v'
    inst✝⁷ : CommRing R'
    inst✝⁶ : CommRing S'
    inst✝⁵ : Algebra R' S'
    P' : Algebra.Generators R' S'
    inst✝⁴ : Algebra R R'
    inst✝³ : Algebra S S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    f : P.Hom P'
    i : P.vars
    j : P'.vars
    ⊢ Eq ((P'.cotangentSpaceBasis.repr (TensorProduct.tmul (Algebra.Extension.mk P …
  -/
  erw [cotangentSpaceBasis_repr_one_tmul, Hom.toAlgHom_X]
  /-
    🎉 no goals
  -/


@[simp]
lemma toKaehler_cotangentSpaceBasis (i) :
    P.toExtension.toKaehler (P.cotangentSpaceBasis i) = D R S (P.val i) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    i : P.vars
    ⊢ Eq (P.toExtension.toKaehler (P.cotangentSpaceBasis i)) ((KaehlerDifferential …
  -/
  rw [cotangentSpaceBasis_apply]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Generators R S
    i : P.vars
    ⊢ Eq (P.toExtension.toKaehler (TensorProduct.tmul P.Ring 1 ((KaehlerDifferenti …
  -/
  exact (KaehlerDifferential.mapBaseChange_tmul ..).trans (by simp)
  /-
    🎉 no goals
  -/


open Extension.H1Cotangent in
/-- `H¹(L_{S/R})` is independent of the presentation chosen. -/
@[simps! apply]
noncomputable
def Generators.H1Cotangent.equiv (P : Generators R S) (P' : Generators R S) :
    P.toExtension.H1Cotangent ≃ₗ[S] P'.toExtension.H1Cotangent where
  __ := map (Generators.defaultHom P P').toExtensionHom
  invFun := map (Generators.defaultHom P' P).toExtensionHom
  left_inv x :=
    show ((map (defaultHom P' P).toExtensionHom) ∘ₗ
      (map (defaultHom P P').toExtensionHom)) x = LinearMap.id x by
    rw [← Extension.H1Cotangent.map_id, eq_comm, map_eq _ ((defaultHom P' P).toExtensionHom.comp
                                                                          /-
                                                                            R : Type u
                                                                            S : Type v
                                                                            inst✝² : CommRing R
                                                                            inst✝¹ : CommRing S
                                                                            inst✝ : Algebra R S
                                                                            P✝ : Algebra.Generators R S
                                                                            P : Algebra.Generators R S
                                                                            P' : Algebra.Generators R S
                                                                            x : P.toExtension.H1Cotangent
                                                                            ⊢ Eq (((↑S (Algebra.Extension.H1Cotangent.map (P'.defaultHom P).toExtensionHom …
                                                                          -/
      (defaultHom P P').toExtensionHom), Extension.H1Cotangent.map_comp]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  right_inv x :=
    show ((map (defaultHom P P').toExtensionHom) ∘ₗ
      (map (defaultHom P' P).toExtensionHom)) x = LinearMap.id x by
    rw [← Extension.H1Cotangent.map_id, eq_comm, map_eq _ ((defaultHom P P').toExtensionHom.comp
                                                                          /-
                                                                            R : Type u
                                                                            S : Type v
                                                                            inst✝² : CommRing R
                                                                            inst✝¹ : CommRing S
                                                                            inst✝ : Algebra R S
                                                                            P✝ : Algebra.Generators R S
                                                                            P : Algebra.Generators R S
                                                                            P' : Algebra.Generators R S
                                                                            x : P'.toExtension.H1Cotangent
                                                                            ⊢ Eq (((↑S (Algebra.Extension.H1Cotangent.map (P.defaultHom P').toExtensionHom …
                                                                          -/
      (defaultHom P' P).toExtensionHom), Extension.H1Cotangent.map_comp]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- `H¹(L_{S/R})`, the first homology of the (naive) cotangent complex of `S` over `R`. -/
abbrev H1Cotangent : Type _ := (Generators.self R S).toExtension.H1Cotangent


/-- The induced map on the first homology of the (naive) cotangent complex of `S` over `R`. -/
noncomputable
def H1Cotangent.map : H1Cotangent R S' →ₗ[S'] H1Cotangent S T :=
  Extension.H1Cotangent.map (Generators.defaultHom _ _).toExtensionHom


/-- `H¹(L_{S/R})` is independent of the presentation chosen. -/
noncomputable
abbrev Generators.equivH1Cotangent (P : Generators.{w} R S) :
    P.toExtension.H1Cotangent ≃ₗ[S] H1Cotangent R S :=
  Generators.H1Cotangent.equiv _ _


