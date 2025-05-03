/-- Any `S`-module M is also an `R`-module via a ring homomorphism `f : R ⟶ S` by defining
    `r • m := f r • m` (`Module.compHom`). This is called restriction of scalars. -/
def obj' : ModuleCat R :=
  let _ := Module.compHom M f
  of R M


/-- Given an `S`-linear map `g : M → M'` between `S`-modules, `g` is also `R`-linear between `M` and
`M'` by means of restriction of scalars.
-/
def map' {M M' : ModuleCat.{v} S} (g : M ⟶ M') : obj' f M ⟶ obj' f M' :=
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(X := ...)` and `(Y := ...)`.
  -- This suggests `RestrictScalars.obj'` needs to be redesigned.
  ofHom (X := obj' f M) (Y := obj' f M')
    { g.hom with map_smul' := fun r => g.hom.map_smul (f r) }


/-- The restriction of scalars operation is functorial. For any `f : R →+* S` a ring homomorphism,
* an `S`-module `M` can be considered as `R`-module by `r • m = f r • m`
* an `S`-linear map is also `R`-linear
-/
def restrictScalars {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    ModuleCat.{v} S ⥤ ModuleCat.{v} R where
  obj := RestrictScalars.obj' f
  map := RestrictScalars.map' f


instance {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    (restrictScalars.{v} f).Faithful where
  map_injective h := by
    /-
      R : Type u₁
      S : Type u₂
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      X✝ Y✝ : ModuleCat S
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((ModuleCat.restrictScalars f).map a₁✝) ((ModuleCat.restrictScalars f). …
      ⊢ Eq a₁✝ a₂✝
    -/
    ext x
    /-
      case hf.h
      R : Type u₁
      S : Type u₂
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      X✝ Y✝ : ModuleCat S
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((ModuleCat.restrictScalars f).map a₁✝) ((ModuleCat.restrictScalars f). …
      x : ↑X✝
      ⊢ Eq (a₁✝.hom x) (a₂✝.hom x)
    -/
    simpa only using DFunLike.congr_fun (ModuleCat.hom_ext_iff.mp h) x
    /-
      🎉 no goals
    -/


instance {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    (restrictScalars.{v} f).PreservesMonomorphisms where
                      /-
                        R : Type u₁
                        S : Type u₂
                        inst✝¹ : Ring R
                        inst✝ : Ring S
                        f : RingHom R S
                        X✝ Y✝ : ModuleCat S
                        x✝ : Quiver.Hom X✝ Y✝
                        h : CategoryTheory.Mono x✝
                        ⊢ CategoryTheory.Mono ((ModuleCat.restrictScalars f).map x✝)
                      -/
  preserves _ h := by rwa [mono_iff_injective] at h ⊢
                      /-
                        🎉 no goals
                      -/

-- Porting note: this should be automatic
-- TODO: this instance gives diamonds if `f : S →+* S`, see `PresheafOfModules.pushforward₀`.
-- The correct solution is probably to define explicit maps between `M` and
-- `(restrictScalars f).obj M`.

instance {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] {f : R →+* S}
    {M : ModuleCat.{v} S} : Module S <| (restrictScalars f).obj M :=
  inferInstanceAs <| Module S M


@[simp]
theorem restrictScalars.map_apply {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S)
    {M M' : ModuleCat.{v} S} (g : M ⟶ M') (x) : (restrictScalars f).map g x = g x :=
  rfl


@[simp]
theorem restrictScalars.smul_def {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S)
    {M : ModuleCat.{v} S} (r : R) (m : (restrictScalars f).obj M) : r • m = f r • show M from m :=
  rfl


theorem restrictScalars.smul_def' {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S)
    {M : ModuleCat.{v} S} (r : R) (m : M) :
    r • (show (restrictScalars f).obj M from m) = f r • m :=
  rfl



instance (priority := 100) sMulCommClass_mk {R : Type u₁} {S : Type u₂} [Ring R] [CommRing S]
    (f : R →+* S) (M : Type v) [I : AddCommGroup M] [Module S M] :
    haveI : SMul R M := (RestrictScalars.obj' f (ModuleCat.of S M)).isModule.toSMul
    SMulCommClass R S M :=
  @SMulCommClass.mk R S M (_) _
                    /-
                      R : Type u₁
                      S : Type u₂
                      inst✝² : Ring R
                      inst✝¹ : CommRing S
                      f : RingHom R S
                      M : Type v
                      I : AddCommGroup M
                      inst✝ : Module S M
                      r : R
                      s : S
                      m : M
                      ⊢ Eq (HSMul.hSMul (f r) (HSMul.hSMul s m)) (HSMul.hSMul s (HSMul.hSMul (f r) m))
                    -/
   fun r s m => (by simp [← mul_smul, mul_comm] : f r • s • m = s • f r • m)
                    /-
                      🎉 no goals
                    -/


/-- Semilinear maps `M →ₛₗ[f] N` identify to
morphisms `M ⟶ (ModuleCat.restrictScalars f).obj N`. -/
@[simps]
def semilinearMapAddEquiv {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S)
    (M : ModuleCat.{v} R) (N : ModuleCat.{v} S) :
    (M →ₛₗ[f] N) ≃+ (M ⟶ (ModuleCat.restrictScalars f).obj N) where
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  toFun g := ofHom (Y := (ModuleCat.restrictScalars f).obj N) <|
    { toFun := g
                     /-
                       R : Type u₁
                       S : Type u₂
                       inst✝¹ : Ring R
                       inst✝ : Ring S
                       f : RingHom R S
                       M : ModuleCat R
                       N : ModuleCat S
                       g : LinearMap f ↑M ↑N
                       ⊢ ∀ (x y : ↑M), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u₁
                        S : Type u₂
                        inst✝¹ : Ring R
                        inst✝ : Ring S
                        f : RingHom R S
                        M : ModuleCat R
                        N : ModuleCat S
                        g : LinearMap f ↑M ↑N
                        ⊢ ∀ (m : R) (x : ↑M), Eq ({ toFun := ⇑g, map_add' := ⋯ }.toFun (HSMul.hSMul m  …
                      -/
      map_smul' := by simp }
                      /-
                        🎉 no goals
                      -/
  invFun g :=
    { toFun := g
                     /-
                       R : Type u₁
                       S : Type u₂
                       inst✝¹ : Ring R
                       inst✝ : Ring S
                       f : RingHom R S
                       M : ModuleCat R
                       N : ModuleCat S
                       g : Quiver.Hom M ((ModuleCat.restrictScalars f).obj N)
                       ⊢ ∀ (x y : ↑M), Eq (g.hom (HAdd.hAdd x y)) (HAdd.hAdd (g.hom x) (g.hom y))
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
      map_smul' := g.hom.map_smul }
  left_inv _ := rfl
  right_inv _ := rfl
  map_add' _ _ := rfl


/-- For a `R`-module `M`, the restriction of scalars of `M` by the identity morphisms identifies
to `M`. -/
def restrictScalarsId'App (hf : f = RingHom.id R) (M : ModuleCat R) :
    (restrictScalars f).obj M ≅ M :=
  LinearEquiv.toModuleIso <|
    @AddEquiv.toLinearEquiv _ _ _ _ _ _ (((restrictScalars f).obj M).isModule) _
          /-
            R : Type u₁
            inst✝ : Ring R
            f : RingHom R R
            hf : Eq f (RingHom.id R)
            M : ModuleCat R
            ⊢ AddEquiv ↑((ModuleCat.restrictScalars f).obj M) ↑M
          -/
          /-
            🎉 no goals
          -/
      (by rfl) (fun r x ↦ by subst hf; rfl)
                                       /-
                                         🎉 no goals
                                       -/


@[simp] lemma restrictScalarsId'App_hom_apply (M : ModuleCat R) (x : M) :
    (restrictScalarsId'App f hf M).hom x = x :=
  rfl


@[simp] lemma restrictScalarsId'App_inv_apply (M : ModuleCat R) (x : M) :
    (restrictScalarsId'App f hf M).inv x = x :=
  rfl


/-- The restriction of scalars by a ring morphism that is the identity identify to the
identity functor. -/
@[simps! hom_app inv_app]
def restrictScalarsId' : ModuleCat.restrictScalars.{v} f ≅ 𝟭 _ :=
    NatIso.ofComponents <| fun M ↦ restrictScalarsId'App f hf M


@[reassoc]
lemma restrictScalarsId'App_hom_naturality {M N : ModuleCat R} (φ : M ⟶ N) :
    (restrictScalars f).map φ ≫ (restrictScalarsId'App f hf N).hom =
      (restrictScalarsId'App f hf M).hom ≫ φ :=
  (restrictScalarsId' f hf).hom.naturality φ


@[reassoc]
lemma restrictScalarsId'App_inv_naturality {M N : ModuleCat R} (φ : M ⟶ N) :
    φ ≫ (restrictScalarsId'App f hf N).inv =
      (restrictScalarsId'App f hf M).inv ≫ (restrictScalars f).map φ :=
  (restrictScalarsId' f hf).inv.naturality φ


/-- The restriction of scalars by the identity morphisms identify to the
identity functor. -/
abbrev restrictScalarsId := restrictScalarsId'.{v} (RingHom.id R) rfl


/-- For each `R₃`-module `M`, restriction of scalars of `M` by a composition of ring morphisms
identifies to successively restricting scalars. -/
def restrictScalarsComp'App (hgf : gf = g.comp f) (M : ModuleCat R₃) :
    (restrictScalars gf).obj M ≅ (restrictScalars f).obj ((restrictScalars g).obj M) :=
  (AddEquiv.toLinearEquiv
    (M := ↑((restrictScalars gf).obj M))
    (M₂ := ↑((restrictScalars f).obj ((restrictScalars g).obj M)))
        /-
          R₁ : Type u₁
          R₂ : Type u₂
          R₃ : Type u₃
          inst✝² : Ring R₁
          inst✝¹ : Ring R₂
          inst✝ : Ring R₃
          f : RingHom R₁ R₂
          g : RingHom R₂ R₃
          gf : RingHom R₁ R₃
          hgf : Eq gf (g.comp f)
          M : ModuleCat R₃
          ⊢ AddEquiv ↑((ModuleCat.restrictScalars gf).obj M) ↑((ModuleCat.restrictScalar …
        -/
    (by rfl)
        /-
          🎉 no goals
        -/
                  /-
                    R₁ : Type u₁
                    R₂ : Type u₂
                    R₃ : Type u₃
                    inst✝² : Ring R₁
                    inst✝¹ : Ring R₂
                    inst✝ : Ring R₃
                    f : RingHom R₁ R₂
                    g : RingHom R₂ R₃
                    gf : RingHom R₁ R₃
                    hgf : Eq gf (g.comp f)
                    M : ModuleCat R₃
                    r : R₁
                    x : ↑((ModuleCat.restrictScalars gf).obj M)
                    ⊢ Eq ((AddEquiv.refl ↑((ModuleCat.restrictScalars gf).obj M)) (HSMul.hSMul r x …
                  -/
    (fun r x ↦ by subst hgf; rfl)).toModuleIso
                             /-
                               🎉 no goals
                             -/


@[simp] lemma restrictScalarsComp'App_hom_apply (M : ModuleCat R₃) (x : M) :
    (restrictScalarsComp'App f g gf hgf M).hom x = x :=
  rfl


@[simp] lemma restrictScalarsComp'App_inv_apply (M : ModuleCat R₃) (x : M) :
    (restrictScalarsComp'App f g gf hgf M).inv x = x :=
  rfl


/-- The restriction of scalars by a composition of ring morphisms identify to the
composition of the restriction of scalars functors. -/
@[simps! hom_app inv_app]
def restrictScalarsComp' :
    ModuleCat.restrictScalars.{v} gf ≅
      ModuleCat.restrictScalars g ⋙ ModuleCat.restrictScalars f :=
  NatIso.ofComponents <| fun M ↦ restrictScalarsComp'App f g gf hgf M


@[reassoc]
lemma restrictScalarsComp'App_hom_naturality {M N : ModuleCat R₃} (φ : M ⟶ N) :
    (restrictScalars gf).map φ ≫ (restrictScalarsComp'App f g gf hgf N).hom =
      (restrictScalarsComp'App f g gf hgf M).hom ≫
        (restrictScalars f).map ((restrictScalars g).map φ) :=
  (restrictScalarsComp' f g gf hgf).hom.naturality φ


@[reassoc]
lemma restrictScalarsComp'App_inv_naturality {M N : ModuleCat R₃} (φ : M ⟶ N) :
    (restrictScalars f).map ((restrictScalars g).map φ) ≫
        (restrictScalarsComp'App f g gf hgf N).inv =
      (restrictScalarsComp'App f g gf hgf M).inv ≫ (restrictScalars gf).map φ :=
  (restrictScalarsComp' f g gf hgf).inv.naturality φ


/-- The restriction of scalars by a composition of ring morphisms identify to the
composition of the restriction of scalars functors. -/
abbrev restrictScalarsComp := restrictScalarsComp'.{v} f g _ rfl


/-- The equivalence of categories `ModuleCat S ≌ ModuleCat R` induced by `e : R ≃+* S`. -/
def restrictScalarsEquivalenceOfRingEquiv {R S} [Ring R] [Ring S] (e : R ≃+* S) :
    ModuleCat S ≌ ModuleCat R where
  functor := ModuleCat.restrictScalars e.toRingHom
  inverse := ModuleCat.restrictScalars e.symm
  unitIso := NatIso.ofComponents (fun M ↦ LinearEquiv.toModuleIso
    (X₁ := M)
    (X₂ := (restrictScalars e.symm.toRingHom).obj ((restrictScalars e.toRingHom).obj M))
    { __ := AddEquiv.refl M
                                                                           /-
                                                                             R : Type ?u.56546
                                                                             S : Type ?u.56549
                                                                             inst✝¹ : Ring R
                                                                             inst✝ : Ring S
                                                                             e : RingEquiv R S
                                                                             ⊢ ∀ {X Y : ModuleCat S} (f : Quiver.Hom X Y),
                                                                                 Eq
                                                                                   (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (ModuleC …
                                                                                     ((fun M =>
                                                                                           (let __spread.0 := AddEquiv.refl ↑M;
                                                                                             { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := ⋯, in …
                                                                                         Y).hom)
                                                                                   (CategoryTheory.CategoryStruct.comp
                                                                                     ((fun M =>
                                                                                           (let __spread.0 := AddEquiv.refl ↑M;
                                                                                             { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := ⋯, in …
                                                                                         X).hom
                                                                                     (((ModuleCat.restrictScalars e.toRingHom).comp (ModuleCat.restrictScal …
                                                                           -/
      map_smul' := fun s m ↦ congr_arg (· • m) (e.right_inv s).symm }) (by intros; rfl)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  counitIso := NatIso.ofComponents (fun M ↦ LinearEquiv.toModuleIso
    (X₁ := (restrictScalars e.toRingHom).obj ((restrictScalars e.symm.toRingHom).obj M))
    (X₂ := M)
    { __ := AddEquiv.refl M
                                                                          /-
                                                                            R : Type ?u.56546
                                                                            S : Type ?u.56549
                                                                            inst✝¹ : Ring R
                                                                            inst✝ : Ring S
                                                                            e : RingEquiv R S
                                                                            ⊢ ∀ {X Y : ModuleCat R} (f : Quiver.Hom X Y),
                                                                                Eq
                                                                                  (CategoryTheory.CategoryStruct.comp (((ModuleCat.restrictScalars ↑e.symm …
                                                                                    ((fun M =>
                                                                                          (let __spread.0 := AddEquiv.refl ↑M;
                                                                                            { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := ⋯, in …
                                                                                        Y).hom)
                                                                                  (CategoryTheory.CategoryStruct.comp
                                                                                    ((fun M =>
                                                                                          (let __spread.0 := AddEquiv.refl ↑M;
                                                                                            { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := ⋯, in …
                                                                                        X).hom
                                                                                    ((CategoryTheory.Functor.id (ModuleCat R)).map f))
                                                                          -/
      map_smul' := fun r _ ↦ congr_arg (· • (_ : M)) (e.left_inv r)}) (by intros; rfl)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                             /-
                               R : Type ?u.56546
                               S : Type ?u.56549
                               inst✝¹ : Ring R
                               inst✝ : Ring S
                               e : RingEquiv R S
                               ⊢ ∀ (X : ModuleCat S),
                                   Eq
                                     (CategoryTheory.CategoryStruct.comp
                                       ((ModuleCat.restrictScalars e.toRingHom).map
                                         ((CategoryTheory.NatIso.ofComponents
                                                 (fun M =>
                                                   (let __spread.0 := AddEquiv.refl ↑M;
                                                     { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := …
                                                 ⋯).hom.app
                                           X))
                                       ((CategoryTheory.NatIso.ofComponents
                                               (fun M =>
                                                 (let __spread.0 := AddEquiv.refl ↑M;
                                                   { toFun := __spread.0.toFun, map_add' := ⋯, map_smul' := ⋯ …
                                               ⋯).hom.app
                                         ((ModuleCat.restrictScalars e.toRingHom).obj X)))
                                     (CategoryTheory.CategoryStruct.id ((ModuleCat.restrictScalars e.toRingHo …
                             -/
  functor_unitIso_comp := by intros; rfl
                                     /-
                                       🎉 no goals
                                     -/


instance restrictScalars_isEquivalence_of_ringEquiv {R S} [Ring R] [Ring S] (e : R ≃+* S) :
    (ModuleCat.restrictScalars e.toRingHom).IsEquivalence :=
  (restrictScalarsEquivalenceOfRingEquiv e).isEquivalence_functor


scoped[ChangeOfRings]
  notation s "⊗ₜ[" R "," f "]" m => @TensorProduct.tmul R _ _ _ _ _ (Module.compHom _ f) _ s m


/-- Extension of scalars turn an `R`-module into `S`-module by M ↦ S ⨂ M
-/
def obj' : ModuleCat S :=
  of _ (TensorProduct R ((restrictScalars f).obj (of _ S)) M)


/-- Extension of scalars is a functor where an `R`-module `M` is sent to `S ⊗ M` and
`l : M1 ⟶ M2` is sent to `s ⊗ m ↦ s ⊗ l m`
-/
def map' {M1 M2 : ModuleCat.{v} R} (l : M1 ⟶ M2) : obj' f M1 ⟶ obj' f M2 :=
  ofHom (@LinearMap.baseChange R S M1 M2 _ _ ((algebraMap S _).comp f).toAlgebra _ _ _ _ l.hom)


theorem map'_id {M : ModuleCat.{v} R} : map' f (𝟙 M) = 𝟙 _ := by
  /-
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    ⊢ Eq (ModuleCat.ExtendScalars.map' f (CategoryTheory.CategoryStruct.id M)) (Ca …
  -/
  ext x
  /-
    case hf.h
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    x : ↑(ModuleCat.ExtendScalars.obj' f M)
    ⊢ Eq ((ModuleCat.ExtendScalars.map' f (CategoryTheory.CategoryStruct.id M)).ho …
  -/
  simp [map']
  /-
    🎉 no goals
  -/


theorem map'_comp {M₁ M₂ M₃ : ModuleCat.{v} R} (l₁₂ : M₁ ⟶ M₂) (l₂₃ : M₂ ⟶ M₃) :
    map' f (l₁₂ ≫ l₂₃) = map' f l₁₂ ≫ map' f l₂₃ := by
  /-
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M₁ M₂ M₃ : ModuleCat R
    l₁₂ : Quiver.Hom M₁ M₂
    l₂₃ : Quiver.Hom M₂ M₃
    ⊢ Eq (ModuleCat.ExtendScalars.map' f (CategoryTheory.CategoryStruct.comp l₁₂ l …
  -/
  ext x
  /-
    case hf.h
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M₁ M₂ M₃ : ModuleCat R
    l₁₂ : Quiver.Hom M₁ M₂
    l₂₃ : Quiver.Hom M₂ M₃
    x : ↑(ModuleCat.ExtendScalars.obj' f M₁)
    ⊢ Eq ((ModuleCat.ExtendScalars.map' f (CategoryTheory.CategoryStruct.comp l₁₂  …
  -/
  dsimp only [map']
  induction x using TensorProduct.induction_on with
  | zero => rfl
  | tmul => rfl
  | add _ _ ihx ihy => rw [map_add, map_add, ihx, ihy]


/-- Extension of scalars is a functor where an `R`-module `M` is sent to `S ⊗ M` and
`l : M1 ⟶ M2` is sent to `s ⊗ m ↦ s ⊗ l m`
-/
def extendScalars {R : Type u₁} {S : Type u₂} [CommRing R] [CommRing S] (f : R →+* S) :
    ModuleCat R ⥤ ModuleCat S where
  obj M := ExtendScalars.obj' f M
  map l := ExtendScalars.map' f l
  map_id _ := ExtendScalars.map'_id f
  map_comp := ExtendScalars.map'_comp f


@[simp]
protected theorem smul_tmul {M : ModuleCat.{v} R} (s s' : S) (m : M) :
    s • (s'⊗ₜ[R,f]m : (extendScalars f).obj M) = (s * s')⊗ₜ[R,f]m :=
  rfl


@[simp]
theorem map_tmul {M M' : ModuleCat.{v} R} (g : M ⟶ M') (s : S) (m : M) :
    (extendScalars f).map g (s⊗ₜ[R,f]m) = s⊗ₜ[R,f]g m :=
  rfl


@[ext]
lemma hom_ext {M : ModuleCat R} {N : ModuleCat S}
    {α β : (extendScalars f).obj M ⟶ N}
    (h : ∀ (m : M), α ((1 : S) ⊗ₜ m) = β ((1 : S) ⊗ₜ m)) : α = β := by
  /-
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    ⊢ Eq α β
  -/
  apply (restrictScalars f).map_injective
  /-
    case a
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    ⊢ Eq ((ModuleCat.restrictScalars f).map α) ((ModuleCat.restrictScalars f).map β)
  -/
  letI := f.toAlgebra
  /-
    case a
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    this : Algebra R S := f.toAlgebra
    ⊢ Eq ((ModuleCat.restrictScalars f).map α) ((ModuleCat.restrictScalars f).map β)
  -/
  ext : 1
  /-
    case a.hf
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    this : Algebra R S := f.toAlgebra
    ⊢ Eq ((ModuleCat.restrictScalars f).map α).hom ((ModuleCat.restrictScalars f). …
  -/
  apply TensorProduct.ext'
  /-
    case a.hf.H
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    this : Algebra R S := f.toAlgebra
    ⊢ ∀ (x : ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S))) (y : ↑M), Eq …
  -/
  intro (s : S) m
  /-
    case a.hf.H
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    this : Algebra R S := f.toAlgebra
    s : S
    m : ↑M
    ⊢ Eq (((ModuleCat.restrictScalars f).map α).hom (TensorProduct.tmul R s m)) (( …
  -/
  change α (s ⊗ₜ m) = β (s ⊗ₜ m)
  have : s ⊗ₜ[R] (m : M) = s • (1 : S) ⊗ₜ[R] m := by
    rw [ExtendScalars.smul_tmul, mul_one]
  /-
    case a.hf.H
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    M : ModuleCat R
    N : ModuleCat S
    α β : Quiver.Hom ((ModuleCat.extendScalars f).obj M) N
    h : ∀ (m : ↑M), Eq (α.hom (TensorProduct.tmul R 1 m)) (β.hom (TensorProduct.tm …
    this✝ : Algebra R S := f.toAlgebra
    s : S
    m : ↑M
    this : Eq (TensorProduct.tmul R s m) (HSMul.hSMul s (TensorProduct.tmul R 1 m))
    ⊢ Eq (α.hom (TensorProduct.tmul R s m)) (β.hom (TensorProduct.tmul R s m))
  -/
  simp only [this, map_smul, h]
  /-
    🎉 no goals
  -/


/-- Given an `R`-module M, consider Hom(S, M) -- the `R`-linear maps between S (as an `R`-module by
 means of restriction of scalars) and M. `S` acts on Hom(S, M) by `s • g = x ↦ g (x • s)`
 -/
instance hasSMul : SMul S <| (restrictScalars f).obj (of _ S) →ₗ[R] M where
  smul s g :=
    { toFun := fun s' : S => g (s' * s : S)
                                    /-
                                      R✝ : Type u₁
                                      S✝ : Type u₂
                                      inst✝⁵ : CommRing R✝
                                      inst✝⁴ : CommRing S✝
                                      f✝ : RingHom R✝ S✝
                                      R : Type u₁
                                      S : Type u₂
                                      inst✝³ : Ring R
                                      inst✝² : Ring S
                                      f : RingHom R S
                                      M : Type v
                                      inst✝¹ : AddCommMonoid M
                                      inst✝ : Module R M
                                      s : S
                                      g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
                                      x y : S
                                      ⊢ Eq ((fun s' => g (HMul.hMul s' s)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun s' => g …
                                    -/
      map_add' := fun x y : S => by dsimp; rw [add_mul, map_add]
                                           /-
                                             🎉 no goals
                                           -/
      map_smul' := fun r (t : S) => by
        -- Porting note: needed some erw's even after dsimp to clean things up
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝⁵ : CommRing R✝
          inst✝⁴ : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝³ : Ring R
          inst✝² : Ring S
          f : RingHom R S
          M : Type v
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          s : S
          g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
          r : R
          t : S
          ⊢ Eq ({ toFun := fun s' => g (HMul.hMul s' s), map_add' := ⋯ }.toFun (HSMul.hS …
        -/
        dsimp
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝⁵ : CommRing R✝
          inst✝⁴ : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝³ : Ring R
          inst✝² : Ring S
          f : RingHom R S
          M : Type v
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          s : S
          g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
          r : R
          t : S
          ⊢ Eq (g (HMul.hMul (HSMul.hSMul r t) s)) (HSMul.hSMul r (g (HMul.hMul t s)))
        -/
        rw [← LinearMap.map_smul]
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝⁵ : CommRing R✝
          inst✝⁴ : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝³ : Ring R
          inst✝² : Ring S
          f : RingHom R S
          M : Type v
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          s : S
          g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
          r : R
          t : S
          ⊢ Eq (g (HMul.hMul (HSMul.hSMul r t) s)) (g (HSMul.hSMul r (HMul.hMul t s)))
        -/
        erw [smul_eq_mul, smul_eq_mul, mul_assoc] }
        /-
          🎉 no goals
        -/


@[simp]
theorem smul_apply' (s : S) (g : (restrictScalars f).obj (of _ S) →ₗ[R] M) (s' : S) :
    (s • g) s' = g (s' * s : S) :=
  rfl


instance mulAction : MulAction S <| (restrictScalars f).obj (of _ S) →ₗ[R] M :=
  { CoextendScalars.hasSMul f _ with
                                                       /-
                                                         R✝ : Type u₁
                                                         S✝ : Type u₂
                                                         inst✝⁵ : CommRing R✝
                                                         inst✝⁴ : CommRing S✝
                                                         f✝ : RingHom R✝ S✝
                                                         R : Type u₁
                                                         S : Type u₂
                                                         inst✝³ : Ring R
                                                         inst✝² : Ring S
                                                         f : RingHom R S
                                                         M : Type v
                                                         inst✝¹ : AddCommMonoid M
                                                         inst✝ : Module R M
                                                         g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
                                                         s : S
                                                         ⊢ Eq ((HSMul.hSMul 1 g) s) (g s)
                                                       -/
    one_smul := fun g => LinearMap.ext fun s : S => by simp
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                                 /-
                                                                   R✝ : Type u₁
                                                                   S✝ : Type u₂
                                                                   inst✝⁵ : CommRing R✝
                                                                   inst✝⁴ : CommRing S✝
                                                                   f✝ : RingHom R✝ S✝
                                                                   R : Type u₁
                                                                   S : Type u₂
                                                                   inst✝³ : Ring R
                                                                   inst✝² : Ring S
                                                                   f : RingHom R S
                                                                   M : Type v
                                                                   inst✝¹ : AddCommMonoid M
                                                                   inst✝ : Module R M
                                                                   s t : S
                                                                   g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
                                                                   x : S
                                                                   ⊢ Eq ((HSMul.hSMul (HMul.hMul s t) g) x) ((HSMul.hSMul s (HSMul.hSMul t g)) x)
                                                                 -/
    mul_smul := fun (s t : S) g => LinearMap.ext fun x : S => by simp [mul_assoc] }
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance distribMulAction : DistribMulAction S <| (restrictScalars f).obj (of _ S) →ₗ[R] M :=
  { CoextendScalars.mulAction f _ with
                                                           /-
                                                             R✝ : Type u₁
                                                             S✝ : Type u₂
                                                             inst✝⁵ : CommRing R✝
                                                             inst✝⁴ : CommRing S✝
                                                             f✝ : RingHom R✝ S✝
                                                             R : Type u₁
                                                             S : Type u₂
                                                             inst✝³ : Ring R
                                                             inst✝² : Ring S
                                                             f : RingHom R S
                                                             M : Type v
                                                             inst✝¹ : AddCommMonoid M
                                                             inst✝ : Module R M
                                                             s : S
                                                             g h : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat …
                                                             x✝ : S
                                                             ⊢ Eq ((HSMul.hSMul s (HAdd.hAdd g h)) x✝) ((HAdd.hAdd (HSMul.hSMul s g) (HSMul …
                                                           -/
                                                        /-
                                                          R✝ : Type u₁
                                                          S✝ : Type u₂
                                                          inst✝⁵ : CommRing R✝
                                                          inst✝⁴ : CommRing S✝
                                                          f✝ : RingHom R✝ S✝
                                                          R : Type u₁
                                                          S : Type u₂
                                                          inst✝³ : Ring R
                                                          inst✝² : Ring S
                                                          f : RingHom R S
                                                          M : Type v
                                                          inst✝¹ : AddCommMonoid M
                                                          inst✝ : Module R M
                                                          x✝¹ x✝ : S
                                                          ⊢ Eq ((HSMul.hSMul x✝¹ 0) x✝) (0 x✝)
                                                        -/
    smul_add := fun s g h => LinearMap.ext fun _ : S => by simp
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    smul_zero := fun _ => LinearMap.ext fun _ : S => by simp }


/-- `S` acts on Hom(S, M) by `s • g = x ↦ g (x • s)`, this action defines an `S`-module structure on
Hom(S, M).
 -/
instance isModule : Module S <| (restrictScalars f).obj (of _ S) →ₗ[R] M :=
  { CoextendScalars.distribMulAction f _ with
                                                             /-
                                                               R✝ : Type u₁
                                                               S✝ : Type u₂
                                                               inst✝⁵ : CommRing R✝
                                                               inst✝⁴ : CommRing S✝
                                                               f✝ : RingHom R✝ S✝
                                                               R : Type u₁
                                                               S : Type u₂
                                                               inst✝³ : Ring R
                                                               inst✝² : Ring S
                                                               f : RingHom R S
                                                               M : Type v
                                                               inst✝¹ : AddCommMonoid M
                                                               inst✝ : Module R M
                                                               s1 s2 : S
                                                               g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
                                                               x : S
                                                               ⊢ Eq ((HSMul.hSMul (HAdd.hAdd s1 s2) g) x) ((HAdd.hAdd (HSMul.hSMul s1 g) (HSM …
                                                             -/
    add_smul := fun s1 s2 g => LinearMap.ext fun x : S => by simp [mul_add, LinearMap.map_add]
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                        /-
                                                          R✝ : Type u₁
                                                          S✝ : Type u₂
                                                          inst✝⁵ : CommRing R✝
                                                          inst✝⁴ : CommRing S✝
                                                          f✝ : RingHom R✝ S✝
                                                          R : Type u₁
                                                          S : Type u₂
                                                          inst✝³ : Ring R
                                                          inst✝² : Ring S
                                                          f : RingHom R S
                                                          M : Type v
                                                          inst✝¹ : AddCommMonoid M
                                                          inst✝ : Module R M
                                                          g : LinearMap (RingHom.id R) (↑((ModuleCat.restrictScalars f).obj (ModuleCat.o …
                                                          x : S
                                                          ⊢ Eq ((HSMul.hSMul 0 g) x) (0 x)
                                                        -/
    zero_smul := fun g => LinearMap.ext fun x : S => by simp [LinearMap.map_zero] }
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- If `M` is an `R`-module, then the set of `R`-linear maps `S →ₗ[R] M` is an `S`-module with
scalar multiplication defined by `s • l := x ↦ l (x • s)`-/
def obj' : ModuleCat S :=
  of _ ((restrictScalars f).obj (of _ S) →ₗ[R] M)


instance : CoeFun (obj' f M) fun _ => S → M :=
  inferInstanceAs <| CoeFun ((restrictScalars f).obj (of _ S) →ₗ[R] M) _


/-- If `M, M'` are `R`-modules, then any `R`-linear map `g : M ⟶ M'` induces an `S`-linear map
`(S →ₗ[R] M) ⟶ (S →ₗ[R] M')` defined by `h ↦ g ∘ h`-/
@[simps]
def map' {M M' : ModuleCat R} (g : M ⟶ M') : obj' f M ⟶ obj' f M' :=
  ofHom
  { toFun := fun h => g.hom.comp h
    map_add' := fun _ _ => LinearMap.comp_add _ _ _
                               /-
                                 R✝ : Type u₁
                                 S✝ : Type u₂
                                 inst✝³ : CommRing R✝
                                 inst✝² : CommRing S✝
                                 f✝ : RingHom R✝ S✝
                                 R : Type u₁
                                 S : Type u₂
                                 inst✝¹ : Ring R
                                 inst✝ : Ring S
                                 f : RingHom R S
                                 M✝ : ModuleCat R
                                 M M' : ModuleCat R
                                 g : Quiver.Hom M M'
                                 s : S
                                 h : LinearMap (RingHom.id R) ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of …
                                 ⊢ Eq ({ toFun := fun h => g.hom.comp h, map_add' := ⋯ }.toFun (HSMul.hSMul s h …
                               -/
    map_smul' := fun s h => by ext; simp }
                                    /-
                                      🎉 no goals
                                    -/


/--
For any rings `R, S` and a ring homomorphism `f : R →+* S`, there is a functor from `R`-module to
`S`-module defined by `M ↦ (S →ₗ[R] M)` where `S` is considered as an `R`-module via restriction of
scalars and `g : M ⟶ M'` is sent to `h ↦ g ∘ h`.
-/
def coextendScalars {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    ModuleCat R ⥤ ModuleCat S where
  obj := CoextendScalars.obj' f
  map := CoextendScalars.map' f
                 /-
                   R✝ : Type u₁
                   S✝ : Type u₂
                   inst✝³ : CommRing R✝
                   inst✝² : CommRing S✝
                   f✝ : RingHom R✝ S✝
                   R : Type u₁
                   S : Type u₂
                   inst✝¹ : Ring R
                   inst✝ : Ring S
                   f : RingHom R S
                   x✝ : ModuleCat R
                   ⊢ Eq ({ obj := ModuleCat.CoextendScalars.obj' f, map := fun {X Y} => ModuleCat …
                 -/
  map_id _ := by ext; rfl
                      /-
                        🎉 no goals
                      -/
                     /-
                       R✝ : Type u₁
                       S✝ : Type u₂
                       inst✝³ : CommRing R✝
                       inst✝² : CommRing S✝
                       f✝ : RingHom R✝ S✝
                       R : Type u₁
                       S : Type u₂
                       inst✝¹ : Ring R
                       inst✝ : Ring S
                       f : RingHom R S
                       X✝ Y✝ Z✝ : ModuleCat R
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := ModuleCat.CoextendScalars.obj' f, map := fun {X Y} => ModuleCat …
                     -/
  map_comp _ _ := by ext; rfl
                          /-
                            🎉 no goals
                          -/


instance (M : ModuleCat R) : CoeFun ((coextendScalars f).obj M) fun _ => S → M :=
  inferInstanceAs <| CoeFun (CoextendScalars.obj' f M) _


theorem smul_apply (M : ModuleCat R) (g : (coextendScalars f).obj M) (s s' : S) :
    (s • g) s' = g (s' * s) :=
  rfl


@[simp]
theorem map_apply {M M' : ModuleCat R} (g : M ⟶ M') (x) (s : S) :
    (coextendScalars f).map g x s = g (x s) :=
  rfl


/-- Given `R`-module X and `S`-module Y, any `g : (restrictScalars f).obj Y ⟶ X`
corresponds to `Y ⟶ (coextendScalars f).obj X` by sending `y ↦ (s ↦ g (s • y))`
-/
def HomEquiv.fromRestriction {X : ModuleCat R} {Y : ModuleCat S}
    (g : (restrictScalars f).obj Y ⟶ X) : Y ⟶ (coextendScalars f).obj X :=
  ofHom
  { toFun := fun y : Y =>
      { toFun := fun s : S => g <| (s • y : Y)
                                        /-
                                          R✝ : Type u₁
                                          S✝ : Type u₂
                                          inst✝³ : CommRing R✝
                                          inst✝² : CommRing S✝
                                          f✝ : RingHom R✝ S✝
                                          R : Type u₁
                                          S : Type u₂
                                          inst✝¹ : Ring R
                                          inst✝ : Ring S
                                          f : RingHom R S
                                          X : ModuleCat R
                                          Y : ModuleCat S
                                          g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
                                          y : ↑Y
                                          s1 s2 : S
                                          ⊢ Eq ((fun s => g.hom (HSMul.hSMul s y)) (HAdd.hAdd s1 s2)) (HAdd.hAdd ((fun s …
                                        -/
        map_add' := fun s1 s2 : S => by simp only [add_smul]; rw [LinearMap.map_add]
                                                              /-
                                                                🎉 no goals
                                                              -/
        map_smul' := fun r (s : S) => by
          -- Porting note: dsimp clears out some rw's but less eager to apply others with Lean 4
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            X : ModuleCat R
            Y : ModuleCat S
            g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
            y : ↑Y
            r : R
            s : S
            ⊢ Eq ({ toFun := fun s => g.hom (HSMul.hSMul s y), map_add' := ⋯ }.toFun (HSMu …
          -/
          dsimp
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            X : ModuleCat R
            Y : ModuleCat S
            g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
            y : ↑Y
            r : R
            s : S
            ⊢ Eq (g.hom (HSMul.hSMul (HSMul.hSMul r s) y)) (HSMul.hSMul r (g.hom (HSMul.hS …
          -/
          rw [← g.hom.map_smul]
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            X : ModuleCat R
            Y : ModuleCat S
            g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
            y : ↑Y
            r : R
            s : S
            ⊢ Eq (g.hom (HSMul.hSMul (HSMul.hSMul r s) y)) (g.hom (HSMul.hSMul r (HSMul.hS …
          -/
          erw [smul_eq_mul, mul_smul]
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            X : ModuleCat R
            Y : ModuleCat S
            g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
            y : ↑Y
            r : R
            s : S
            ⊢ Eq (g.hom (HSMul.hSMul (↑f.toMonoidWithZeroHom r) (HSMul.hSMul s y))) (g.hom …
          -/
          rfl }
          /-
            🎉 no goals
          -/
    map_add' := fun y1 y2 : Y =>
      LinearMap.ext fun s : S => by
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          Y : ModuleCat S
          g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
          y1 y2 : ↑Y
          s : S
          ⊢ Eq (((fun y => { toFun := fun s => g.hom (HSMul.hSMul s y), map_add' := ⋯, m …
        -/
        simp [smul_add, map_add]
        /-
          🎉 no goals
        -/
    map_smul' := fun (s : S) (y : Y) => LinearMap.ext fun t : S => by
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          Y : ModuleCat S
          g : Quiver.Hom ((ModuleCat.restrictScalars f).obj Y) X
          s : S
          y : ↑Y
          t : S
          ⊢ Eq (({ toFun := fun y => { toFun := fun s => g.hom (HSMul.hSMul s y), map_ad …
        -/
        simp [mul_smul] }
        /-
          🎉 no goals
        -/


/-- This should be autogenerated by `@[simps]` but we need to give `s` the correct type here. -/
@[simp] lemma HomEquiv.fromRestriction_hom_apply_apply {X : ModuleCat R} {Y : ModuleCat S}
    (g : (restrictScalars f).obj Y ⟶ X) (y) (s : S) :
    (HomEquiv.fromRestriction f g).hom y s = g (s • y) := rfl


/-- Given `R`-module X and `S`-module Y, any `g : Y ⟶ (coextendScalars f).obj X`
corresponds to `(restrictScalars f).obj Y ⟶ X` by `y ↦ g y 1`
-/
def HomEquiv.toRestriction {X Y} (g : Y ⟶ (coextendScalars f).obj X) :
    (restrictScalars f).obj Y ⟶ X :=
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(X := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  ofHom (X := (restrictScalars f).obj Y)
  { toFun := fun y : Y => (g y) (1 : S)
                              /-
                                R✝ : Type u₁
                                S✝ : Type u₂
                                inst✝³ : CommRing R✝
                                inst✝² : CommRing S✝
                                f✝ : RingHom R✝ S✝
                                R : Type u₁
                                S : Type u₂
                                inst✝¹ : Ring R
                                inst✝ : Ring S
                                f : RingHom R S
                                X : ModuleCat R
                                Y : ModuleCat S
                                g : Quiver.Hom Y ((ModuleCat.coextendScalars f).obj X)
                                x y : ↑((ModuleCat.restrictScalars f).obj Y)
                                ⊢ Eq ((fun y => (g.hom y) 1) (HAdd.hAdd x y)) (HAdd.hAdd ((fun y => (g.hom y)  …
                              -/
    map_add' := fun x y => by dsimp; rw [g.hom.map_add, LinearMap.add_apply]
                                     /-
                                       🎉 no goals
                                     -/
    map_smul' := fun r (y : Y) => by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom Y ((ModuleCat.coextendScalars f).obj X)
        r : R
        y : ↑Y
        ⊢ Eq ({ toFun := fun y => (g.hom y) 1, map_add' := ⋯ }.toFun (HSMul.hSMul r y) …
      -/
      dsimp
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom Y ((ModuleCat.coextendScalars f).obj X)
        r : R
        y : ↑Y
        ⊢ Eq ((g.hom (HSMul.hSMul (f r) y)) 1) (HSMul.hSMul r ((g.hom y) 1))
      -/
      rw [← LinearMap.map_smul]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom Y ((ModuleCat.coextendScalars f).obj X)
        r : R
        y : ↑Y
        ⊢ Eq ((g.hom (HSMul.hSMul (f r) y)) 1) ((g.hom y) (HSMul.hSMul r 1))
      -/
      erw [smul_eq_mul, mul_one, LinearMap.map_smul]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom Y ((ModuleCat.coextendScalars f).obj X)
        r : R
        y : ↑Y
        ⊢ Eq ((HSMul.hSMul (f r) (g.hom y)) 1) ((g.hom y) (↑f.toMonoidWithZeroHom r))
      -/
      rw [CoextendScalars.smul_apply (s := f r) (g := g y) (s' := 1), one_mul]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom Y ((ModuleCat.coextendScalars f).obj X)
        r : R
        y : ↑Y
        ⊢ Eq ((g.hom y) (f r)) ((g.hom y) (↑f.toMonoidWithZeroHom r))
      -/
      simp }
      /-
        🎉 no goals
      -/


/-- This should be autogenerated by `@[simps]` but we need to give `1` the correct type here. -/
@[simp] lemma HomEquiv.toRestriction_hom_apply {X : ModuleCat R} {Y : ModuleCat S}
    (g : Y ⟶ (coextendScalars f).obj X) (y) :
    (HomEquiv.toRestriction f g).hom y = g.hom y (1 : S) := rfl

-- Porting note: add to address timeout in unit'

/-- Auxiliary definition for `unit'` -/
def app' (Y : ModuleCat S) : Y →ₗ[S] (restrictScalars f ⋙ coextendScalars f).obj Y :=
  { toFun := fun y : Y =>
      { toFun := fun s : S => (s • y : Y)
        map_add' := fun _ _ => add_smul _ _ _
        map_smul' := fun r (s : S) => by
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            Y : ModuleCat S
            y : ↑Y
            r : R
            s : S
            ⊢ Eq ({ toFun := fun s => HSMul.hSMul s y, map_add' := ⋯ }.toFun (HSMul.hSMul  …
          -/
          dsimp only [AddHom.toFun_eq_coe, AddHom.coe_mk, RingHom.id_apply]
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            Y : ModuleCat S
            y : ↑Y
            r : R
            s : S
            ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) y) (HSMul.hSMul r (HSMul.hSMul s y))
          -/
          erw [smul_eq_mul, mul_smul]
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : Ring R
            inst✝ : Ring S
            f : RingHom R S
            Y : ModuleCat S
            y : ↑Y
            r : R
            s : S
            ⊢ Eq (HSMul.hSMul (↑f.toMonoidWithZeroHom r) (HSMul.hSMul s y)) (HSMul.hSMul r …
          -/
          simp }
          /-
            🎉 no goals
          -/
    map_add' := fun y1 y2 =>
      LinearMap.ext fun s : S => by
        -- Porting note: double dsimp seems odd
        dsimp only [AddHom.toFun_eq_coe, AddHom.coe_mk, RingHom.id_apply,
          RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe,
          MonoidHom.coe_coe, ZeroHom.coe_mk, smul_eq_mul, id_eq, eq_mpr_eq_cast, cast_eq,
          Functor.comp_obj]
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          Y : ModuleCat S
          y1 y2 : ↑Y
          s : S
          ⊢ Eq ({ toFun := fun s => HSMul.hSMul s (HAdd.hAdd y1 y2), map_add' := ⋯, map_ …
        -/
        rw [LinearMap.add_apply, LinearMap.coe_mk, LinearMap.coe_mk, LinearMap.coe_mk]
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          Y : ModuleCat S
          y1 y2 : ↑Y
          s : S
          ⊢ Eq ({ toFun := fun s => HSMul.hSMul s (HAdd.hAdd y1 y2), map_add' := ⋯ } s)  …
        -/
        dsimp
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          Y : ModuleCat S
          y1 y2 : ↑Y
          s : S
          ⊢ Eq (HSMul.hSMul s (HAdd.hAdd y1 y2)) (HAdd.hAdd (HSMul.hSMul s y1) (HSMul.hS …
        -/
        rw [smul_add]
        /-
          🎉 no goals
        -/
    map_smul' := fun s (y : Y) => LinearMap.ext fun t : S => by
      -- Porting note: used to be simp [mul_smul]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        Y : ModuleCat S
        s : S
        y : ↑Y
        t : S
        ⊢ Eq (({ toFun := fun y => { toFun := fun s => HSMul.hSMul s y, map_add' := ⋯, …
      -/
      rw [RingHom.id_apply, LinearMap.coe_mk, CoextendScalars.smul_apply', LinearMap.coe_mk]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        Y : ModuleCat S
        s : S
        y : ↑Y
        t : S
        ⊢ Eq ({ toFun := fun s_1 => HSMul.hSMul s_1 (HSMul.hSMul s y), map_add' := ⋯ } …
      -/
      dsimp
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        Y : ModuleCat S
        s : S
        y : ↑Y
        t : S
        ⊢ Eq (HSMul.hSMul t (HSMul.hSMul s y)) (HSMul.hSMul (HMul.hMul t s) y)
      -/
      rw [mul_smul] }
      /-
        🎉 no goals
      -/


/--
The natural transformation from identity functor to the composition of restriction and coextension
of scalars.
-/
@[simps]
protected def unit' : 𝟭 (ModuleCat S) ⟶ restrictScalars f ⋙ coextendScalars f where
  app Y := ofHom (app' f Y)
  naturality Y Y' g :=
    hom_ext <| LinearMap.ext fun y : Y => LinearMap.ext fun s : S => by
      -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10745): previously simp [CoextendScalars.map_apply]
      simp only [ModuleCat.hom_comp, Functor.id_map, Functor.id_obj, Functor.comp_obj,
        Functor.comp_map, LinearMap.coe_comp, Function.comp, CoextendScalars.map_apply,
        restrictScalars.map_apply f]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        Y Y' : ModuleCat S
        g : Quiver.Hom Y Y'
        y : ↑Y
        s : S
        ⊢ Eq (((ModuleCat.RestrictionCoextensionAdj.app' f Y') (g.hom y)) s) (g.hom (( …
      -/
      change s • (g y) = g (s • y)
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        Y Y' : ModuleCat S
        g : Quiver.Hom Y Y'
        y : ↑Y
        s : S
        ⊢ Eq (HSMul.hSMul s (g.hom y)) (g.hom (HSMul.hSMul s y))
      -/
      rw [map_smul]
      /-
        🎉 no goals
      -/


/-- The natural transformation from the composition of coextension and restriction of scalars to
identity functor.
-/
@[simps]
protected def counit' : coextendScalars f ⋙ restrictScalars f ⟶ 𝟭 (ModuleCat R) where
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(X := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  app X := ofHom (X := (restrictScalars f).obj ((coextendScalars f).obj X))
    { toFun := fun g => g.toFun (1 : S)
      map_add' := fun x1 x2 => by
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          x1 x2 : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj …
          ⊢ Eq ((fun g => g.toFun 1) (HAdd.hAdd x1 x2)) (HAdd.hAdd ((fun g => g.toFun 1) …
        -/
        dsimp
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          x1 x2 : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj …
          ⊢ Eq ((HAdd.hAdd x1 x2) 1) (HAdd.hAdd (x1 1) (x2 1))
        -/
        rw [LinearMap.add_apply]
        /-
          🎉 no goals
        -/
      map_smul' := fun r (g : (restrictScalars f).obj ((coextendScalars f).obj X)) => by
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          r : R
          g : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj X))
          ⊢ Eq ({ toFun := fun g => g.toFun 1, map_add' := ⋯ }.toFun (HSMul.hSMul r g))  …
        -/
        dsimp
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          r : R
          g : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj X))
          ⊢ Eq ((HSMul.hSMul (f r) g) 1) (HSMul.hSMul r (g 1))
        -/
        rw [CoextendScalars.smul_apply, one_mul, ← LinearMap.map_smul]
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          r : R
          g : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj X))
          ⊢ Eq (g (f r)) (g (HSMul.hSMul r 1))
        -/
        congr
        /-
          case h.e_6.h
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          r : R
          g : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj X))
          ⊢ Eq (f r) (HSMul.hSMul r 1)
        -/
        change f r = (f r) • (1 : S)
        /-
          case h.e_6.h
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : Ring R
          inst✝ : Ring S
          f : RingHom R S
          X : ModuleCat R
          r : R
          g : ↑((ModuleCat.restrictScalars f).obj ((ModuleCat.coextendScalars f).obj X))
          ⊢ Eq (f r) (HSMul.hSMul (f r) 1)
        -/
        rw [smul_eq_mul (a := f r) (a' := 1), mul_one] }
        /-
          🎉 no goals
        -/


/-- Restriction of scalars is left adjoint to coextension of scalars. -/
-- @[simps] Porting note: not in normal form and not used
def restrictCoextendScalarsAdj {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    restrictScalars.{max v u₂,u₁,u₂} f ⊣ coextendScalars f :=
  Adjunction.mk' {
    homEquiv := fun X Y ↦
      { toFun := RestrictionCoextensionAdj.HomEquiv.fromRestriction.{u₁,u₂,v} f
        invFun := RestrictionCoextensionAdj.HomEquiv.toRestriction.{u₁,u₂,v} f
                                /-
                                  R✝ : Type u₁
                                  S✝ : Type u₂
                                  inst✝³ : CommRing R✝
                                  inst✝² : CommRing S✝
                                  f✝ : RingHom R✝ S✝
                                  R : Type u₁
                                  S : Type u₂
                                  inst✝¹ : Ring R
                                  inst✝ : Ring S
                                  f : RingHom R S
                                  X : ModuleCat S
                                  Y : ModuleCat R
                                  g : Quiver.Hom ((ModuleCat.restrictScalars f).obj X) Y
                                  ⊢ Eq (ModuleCat.RestrictionCoextensionAdj.HomEquiv.toRestriction f (ModuleCat. …
                                -/
        left_inv := fun g => by ext; simp
                                     /-
                                       🎉 no goals
                                     -/
        right_inv := fun g => hom_ext <| LinearMap.ext fun x => LinearMap.ext fun s : S => by
          -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10745): once just simp
          rw [RestrictionCoextensionAdj.HomEquiv.fromRestriction_hom_apply_apply,
              RestrictionCoextensionAdj.HomEquiv.toRestriction_hom_apply, LinearMap.map_smulₛₗ,
              RingHom.id_apply, CoextendScalars.smul_apply', one_mul] }
    unit := RestrictionCoextensionAdj.unit'.{u₁,u₂,v} f
    counit := RestrictionCoextensionAdj.counit'.{u₁,u₂,v} f
    homEquiv_unit := hom_ext <| LinearMap.ext fun _ => rfl
    homEquiv_counit := fun {X Y g} => by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat S
        Y : ModuleCat R
        g : Quiver.Hom X ((ModuleCat.coextendScalars f).obj Y)
        ⊢ Eq (((fun X Y => { toFun := ModuleCat.RestrictionCoextensionAdj.HomEquiv.fro …
      -/
      ext
      /-
        case hf.h
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        X : ModuleCat S
        Y : ModuleCat R
        g : Quiver.Hom X ((ModuleCat.coextendScalars f).obj Y)
        x✝ : ↑((ModuleCat.restrictScalars f).obj X)
        ⊢ Eq ((((fun X Y => { toFun := ModuleCat.RestrictionCoextensionAdj.HomEquiv.fr …
      -/
      simp [RestrictionCoextensionAdj.counit'] }
      /-
        🎉 no goals
      -/


instance {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    (restrictScalars.{max u₂ w} f).IsLeftAdjoint  :=
  (restrictCoextendScalarsAdj f).isLeftAdjoint


instance {R : Type u₁} {S : Type u₂} [Ring R] [Ring S] (f : R →+* S) :
    (coextendScalars.{u₁, u₂, max u₂ w} f).IsRightAdjoint  :=
  (restrictCoextendScalarsAdj f).isRightAdjoint


/--
Given `R`-module X and `S`-module Y and a map `g : (extendScalars f).obj X ⟶ Y`, i.e. `S`-linear
map `S ⨂ X → Y`, there is a `X ⟶ (restrictScalars f).obj Y`, i.e. `R`-linear map `X ⟶ Y` by
`x ↦ g (1 ⊗ x)`.
-/
@[simps hom_apply]
def HomEquiv.toRestrictScalars {X Y} (g : (extendScalars f).obj X ⟶ Y) :
    X ⟶ (restrictScalars f).obj Y :=
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  ofHom (Y := (restrictScalars f).obj Y)
  { toFun := fun x => g <| (1 : S)⊗ₜ[R,f]x
                              /-
                                R✝ : Type u₁
                                S✝ : Type u₂
                                inst✝³ : CommRing R✝
                                inst✝² : CommRing S✝
                                f✝ : RingHom R✝ S✝
                                R : Type u₁
                                S : Type u₂
                                inst✝¹ : CommRing R
                                inst✝ : CommRing S
                                f : RingHom R S
                                X : ModuleCat R
                                Y : ModuleCat S
                                g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
                                x✝¹ x✝ : ↑X
                                ⊢ Eq ((fun x => g.hom (TensorProduct.tmul R 1 x)) (HAdd.hAdd x✝¹ x✝)) (HAdd.hA …
                              -/
    map_add' := fun _ _ => by dsimp; rw [tmul_add, map_add]
                                     /-
                                       🎉 no goals
                                     -/
    map_smul' := fun r s => by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        r : R
        s : ↑X
        ⊢ Eq ({ toFun := fun x => g.hom (TensorProduct.tmul R 1 x), map_add' := ⋯ }.to …
      -/
      letI : Module R S := Module.compHom S f
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        r : R
        s : ↑X
        this : Module R S := Module.compHom S f
        ⊢ Eq ({ toFun := fun x => g.hom (TensorProduct.tmul R 1 x), map_add' := ⋯ }.to …
      -/
      letI : Module R Y := Module.compHom Y f
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        r : R
        s : ↑X
        this✝ : Module R S := Module.compHom S f
        this : Module R ↑Y := Module.compHom (↑Y) f
        ⊢ Eq ({ toFun := fun x => g.hom (TensorProduct.tmul R 1 x), map_add' := ⋯ }.to …
      -/
      dsimp
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        r : R
        s : ↑X
        this✝ : Module R S := Module.compHom S f
        this : Module R ↑Y := Module.compHom (↑Y) f
        ⊢ Eq (g.hom (TensorProduct.tmul R 1 (HSMul.hSMul r s))) (HSMul.hSMul (f r) (g. …
      -/
      erw [RestrictScalars.smul_def, ← LinearMap.map_smul, tmul_smul]
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        r : R
        s : ↑X
        this✝ : Module R S := Module.compHom S f
        this : Module R ↑Y := Module.compHom (↑Y) f
        ⊢ Eq (g.hom (HSMul.hSMul ((algebraMap R R) r) (TensorProduct.tmul R 1 ((Restri …
      -/
      congr }
      /-
        🎉 no goals
      -/

-- Porting note: forced to break apart fromExtendScalars due to timeouts

/--
The map `S → X →ₗ[R] Y` given by `fun s x => s • (g x)`
-/
@[simps]
def HomEquiv.evalAt {X : ModuleCat R} {Y : ModuleCat S} (s : S)
    (g : X ⟶ (restrictScalars f).obj Y) : have : Module R Y := Module.compHom Y f
    X →ₗ[R] Y :=
  @LinearMap.mk _ _ _ _ (RingHom.id R) X Y _ _ _ (_)
    { toFun := fun x => s • (g x : Y)
      map_add' := by
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : CommRing R
          inst✝ : CommRing S
          f : RingHom R S
          X : ModuleCat R
          Y : ModuleCat S
          s : S
          g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
          ⊢ ∀ (x y : ↑X), Eq ((fun x => HSMul.hSMul s (g.hom x)) (HAdd.hAdd x y)) (HAdd. …
        -/
        intros
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : CommRing R
          inst✝ : CommRing S
          f : RingHom R S
          X : ModuleCat R
          Y : ModuleCat S
          s : S
          g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
          x✝ y✝ : ↑X
          ⊢ Eq ((fun x => HSMul.hSMul s (g.hom x)) (HAdd.hAdd x✝ y✝)) (HAdd.hAdd ((fun x …
        -/
        dsimp only
        /-
          R✝ : Type u₁
          S✝ : Type u₂
          inst✝³ : CommRing R✝
          inst✝² : CommRing S✝
          f✝ : RingHom R✝ S✝
          R : Type u₁
          S : Type u₂
          inst✝¹ : CommRing R
          inst✝ : CommRing S
          f : RingHom R S
          X : ModuleCat R
          Y : ModuleCat S
          s : S
          g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
          x✝ y✝ : ↑X
          ⊢ Eq (HSMul.hSMul s (g.hom (HAdd.hAdd x✝ y✝))) (HAdd.hAdd (HSMul.hSMul s (g.ho …
        -/
        rw [map_add, smul_add] }
        /-
          🎉 no goals
        -/
    (by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        s : S
        g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
        ⊢ ∀ (m : R) (x : ↑X), Eq ({ toFun := fun x => HSMul.hSMul s (g.hom x), map_add …
      -/
      intros r x
      rw [AddHom.toFun_eq_coe, AddHom.coe_mk, RingHom.id_apply,
        LinearMap.map_smul, smul_comm r s (g x : Y)] )


/--
Given `R`-module X and `S`-module Y and a map `X ⟶ (restrictScalars f).obj Y`, i.e `R`-linear map
`X ⟶ Y`, there is a map `(extend_scalars f).obj X ⟶ Y`, i.e `S`-linear map `S ⨂ X → Y` by
`s ⊗ x ↦ s • g x`.
-/
@[simps hom_apply]
def HomEquiv.fromExtendScalars {X Y} (g : X ⟶ (restrictScalars f).obj Y) :
    (extendScalars f).obj X ⟶ Y := by
  /-
    R✝ : Type u₁
    S✝ : Type u₂
    inst✝³ : CommRing R✝
    inst✝² : CommRing S✝
    f✝ : RingHom R✝ S✝
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    X : ModuleCat R
    Y : ModuleCat S
    g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
    ⊢ Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
  -/
  letI m1 : Module R S := Module.compHom S f; letI m2 : Module R Y := Module.compHom Y f
  /-
    R✝ : Type u₁
    S✝ : Type u₂
    inst✝³ : CommRing R✝
    inst✝² : CommRing S✝
    f✝ : RingHom R✝ S✝
    R : Type u₁
    S : Type u₂
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    X : ModuleCat R
    Y : ModuleCat S
    g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
    m1 : Module R S := Module.compHom S f
    m2 : Module R ↑Y := Module.compHom (↑Y) f
    ⊢ Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
  -/
  refine ofHom {toFun := fun z => TensorProduct.lift ?_ z, map_add' := ?_, map_smul' := ?_}
  · refine
    {toFun := fun s => HomEquiv.evalAt f s g, map_add' := fun (s₁ s₂ : S) => ?_,
      map_smul' := fun (r : R) (s : S) => ?_}
      /-
        case refine_1.refine_1
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
        m1 : Module R S := Module.compHom S f
        m2 : Module R ↑Y := Module.compHom (↑Y) f
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑X
        s₁ s₂ : S
        ⊢ Eq ((fun s => ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.evalAt f s g) (HAd …
      -/
    · ext
      /-
        case refine_1.refine_1.h
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
        m1 : Module R S := Module.compHom S f
        m2 : Module R ↑Y := Module.compHom (↑Y) f
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑X
        s₁ s₂ : S
        x✝ : ↑X
        ⊢ Eq (((fun s => ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.evalAt f s g) (HA …
      -/
      dsimp only [m2, evalAt_apply, LinearMap.add_apply]
      /-
        case refine_1.refine_1.h
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
        m1 : Module R S := Module.compHom S f
        m2 : Module R ↑Y := Module.compHom (↑Y) f
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑X
        s₁ s₂ : S
        x✝ : ↑X
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd s₁ s₂) (g.hom x✝)) (HAdd.hAdd (HSMul.hSMul s₁ (g. …
      -/
      rw [← add_smul]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
        m1 : Module R S := Module.compHom S f
        m2 : Module R ↑Y := Module.compHom (↑Y) f
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑X
        r : R
        s : S
        ⊢ Eq ({ toFun := fun s => ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.evalAt f …
      -/
    · ext x
      /-
        case refine_1.refine_2.h
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
        m1 : Module R S := Module.compHom S f
        m2 : Module R ↑Y := Module.compHom (↑Y) f
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑X
        r : R
        s : S
        x : ↑X
        ⊢ Eq (({ toFun := fun s => ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.evalAt  …
      -/
      apply mul_smul (f r) s (g x)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      ⊢ ∀ (x y : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S …
    -/
  · intros z₁ z₂
    /-
      case refine_2
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      z₁ z₂ : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S) …
      ⊢ Eq ((fun z => (TensorProduct.lift { toFun := fun s => ModuleCat.ExtendRestri …
    -/
    change lift _ (z₁ + z₂) = lift _ z₁ + lift _ z₂
    /-
      case refine_2
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      z₁ z₂ : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S) …
      ⊢ Eq ((TensorProduct.lift { toFun := fun s => ModuleCat.ExtendRestrictScalarsA …
    -/
    rw [map_add]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      ⊢ ∀ (m : S) (x : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCa …
    -/
  · intro s z
    /-
      case refine_3
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      s : S
      z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑X
      ⊢ Eq ({ toFun := fun z => (TensorProduct.lift { toFun := fun s => ModuleCat.Ex …
    -/
    change lift _ (s • z) = s • lift _ z
    induction z using TensorProduct.induction_on with
    | zero => rw [smul_zero, map_zero, smul_zero]
    | tmul s' x =>
      rw [LinearMap.coe_mk, ExtendScalars.smul_tmul]
      erw [lift.tmul, lift.tmul]
      set s' : S := s'
      change (s * s') • (g x) = s • s' • (g x)
      rw [mul_smul]
    | add _ _ ih1 ih2 => rw [smul_add, map_add, ih1, ih2, map_add, smul_add]


/-- Given `R`-module X and `S`-module Y, `S`-linear linear maps `(extendScalars f).obj X ⟶ Y`
bijectively correspond to `R`-linear maps `X ⟶ (restrictScalars f).obj Y`.
-/
@[simps symm_apply]
def homEquiv {X Y} :
    ((extendScalars f).obj X ⟶ Y) ≃ (X ⟶ (restrictScalars.{max v u₂,u₁,u₂} f).obj Y) where
  toFun := HomEquiv.toRestrictScalars.{u₁,u₂,v} f
  invFun := HomEquiv.fromExtendScalars.{u₁,u₂,v} f
  left_inv g := by
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
      ⊢ Eq (ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.fromExtendScalars f (ModuleC …
    -/
    letI m1 : Module R S := Module.compHom S f; letI m2 : Module R Y := Module.compHom Y f
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      ⊢ Eq (ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.fromExtendScalars f (ModuleC …
    -/
    apply hom_ext
    /-
      case hf
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      ⊢ Eq (ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.fromExtendScalars f (ModuleC …
    -/
    apply LinearMap.ext; intro z
    induction z using TensorProduct.induction_on with
    | zero => rw [map_zero, map_zero]
    | tmul x s =>
      erw [TensorProduct.lift.tmul]
      simp only [LinearMap.coe_mk]
      change S at x
      dsimp
      erw [← LinearMap.map_smul, ExtendScalars.smul_tmul, mul_one x]
      rfl
    | add _ _ ih1 ih2 => rw [map_add, map_add, ih1, ih2]
  right_inv g := by
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      ⊢ Eq (ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.toRestrictScalars f (ModuleC …
    -/
    letI m1 : Module R S := Module.compHom S f; letI m2 : Module R Y := Module.compHom Y f
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      ⊢ Eq (ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.toRestrictScalars f (ModuleC …
    -/
    ext x
    /-
      case hf.h
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      x : ↑X
      ⊢ Eq ((ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.toRestrictScalars f (Module …
    -/
    rw [HomEquiv.toRestrictScalars_hom_apply]
    -- This needs to be `erw` because of some unfolding in `fromExtendScalars`
    /-
      case hf.h
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      x : ↑X
      ⊢ Eq ((ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.fromExtendScalars f g).hom  …
    -/
    erw [HomEquiv.fromExtendScalars_hom_apply]
    /-
      case hf.h
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      x : ↑X
      ⊢ Eq ((TensorProduct.lift { toFun := fun s => ModuleCat.ExtendRestrictScalarsA …
    -/
    rw [lift.tmul, LinearMap.coe_mk, LinearMap.coe_mk]
    /-
      case hf.h
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      x : ↑X
      ⊢ Eq (({ toFun := fun s => ModuleCat.ExtendRestrictScalarsAdj.HomEquiv.evalAt  …
    -/
    dsimp
    /-
      case hf.h
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      X : ModuleCat R
      Y : ModuleCat S
      g : Quiver.Hom X ((ModuleCat.restrictScalars f).obj Y)
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      x : ↑X
      ⊢ Eq (HSMul.hSMul 1 (g.hom x)) (g.hom x)
    -/
    rw [one_smul]
    /-
      🎉 no goals
    -/


/--
For any `R`-module X, there is a natural `R`-linear map from `X` to `X ⨂ S` by sending `x ↦ x ⊗ 1`
-/
-- @[simps] Porting note: not in normal form and not used
def Unit.map {X} : X ⟶ (extendScalars f ⋙ restrictScalars f).obj X :=
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  ofHom (Y := (extendScalars f ⋙ restrictScalars f).obj X)
  { toFun := fun x => (1 : S)⊗ₜ[R,f]x
                               /-
                                 R✝ : Type u₁
                                 S✝ : Type u₂
                                 inst✝³ : CommRing R✝
                                 inst✝² : CommRing S✝
                                 f✝ : RingHom R✝ S✝
                                 R : Type u₁
                                 S : Type u₂
                                 inst✝¹ : CommRing R
                                 inst✝ : CommRing S
                                 f : RingHom R S
                                 X : ModuleCat R
                                 x x' : ↑X
                                 ⊢ Eq ((fun x => TensorProduct.tmul R 1 x) (HAdd.hAdd x x')) (HAdd.hAdd ((fun x …
                               -/
    map_add' := fun x x' => by dsimp; rw [TensorProduct.tmul_add]
                                      /-
                                        🎉 no goals
                                      -/
    map_smul' := fun r x => by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        r : R
        x : ↑X
        ⊢ Eq ({ toFun := fun x => TensorProduct.tmul R 1 x, map_add' := ⋯ }.toFun (HSM …
      -/
      letI m1 : Module R S := Module.compHom S f
      -- Porting note: used to be rfl
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        r : R
        x : ↑X
        m1 : Module R S := Module.compHom S f
        ⊢ Eq ({ toFun := fun x => TensorProduct.tmul R 1 x, map_add' := ⋯ }.toFun (HSM …
      -/
      dsimp; rw [← TensorProduct.smul_tmul,TensorProduct.smul_tmul'] }
             /-
               🎉 no goals
             -/


/--
The natural transformation from identity functor on `R`-module to the composition of extension and
restriction of scalars.
-/
@[simps]
def unit : 𝟭 (ModuleCat R) ⟶ extendScalars f ⋙ restrictScalars.{max v u₂,u₁,u₂} f where
  app _ := Unit.map.{u₁,u₂,v} f


/-- For any `S`-module Y, there is a natural `R`-linear map from `S ⨂ Y` to `Y` by
`s ⊗ y ↦ s • y` -/
@[simps hom_apply]
def Counit.map {Y} : (restrictScalars f ⋙ extendScalars f).obj Y ⟶ Y :=
  ofHom
  { toFun :=
      letI m1 : Module R S := Module.compHom S f
      letI m2 : Module R Y := Module.compHom Y f
      TensorProduct.lift
      { toFun := fun s : S =>
        { toFun := fun y : Y => s • y,
          map_add' := smul_add _
          map_smul' := fun r y => by
            /-
              R✝ : Type u₁
              S✝ : Type u₂
              inst✝³ : CommRing R✝
              inst✝² : CommRing S✝
              f✝ : RingHom R✝ S✝
              R : Type u₁
              S : Type u₂
              inst✝¹ : CommRing R
              inst✝ : CommRing S
              f : RingHom R S
              Y : ModuleCat S
              m1 : Module R S := Module.compHom S f
              m2 : Module R ↑Y := Module.compHom (↑Y) f
              s : S
              r : R
              y : ↑Y
              ⊢ Eq ({ toFun := fun y => HSMul.hSMul s y, map_add' := ⋯ }.toFun (HSMul.hSMul  …
            -/
            change s • f r • y = f r • s • y
            /-
              R✝ : Type u₁
              S✝ : Type u₂
              inst✝³ : CommRing R✝
              inst✝² : CommRing S✝
              f✝ : RingHom R✝ S✝
              R : Type u₁
              S : Type u₂
              inst✝¹ : CommRing R
              inst✝ : CommRing S
              f : RingHom R S
              Y : ModuleCat S
              m1 : Module R S := Module.compHom S f
              m2 : Module R ↑Y := Module.compHom (↑Y) f
              s : S
              r : R
              y : ↑Y
              ⊢ Eq (HSMul.hSMul s (HSMul.hSMul (f r) y)) (HSMul.hSMul (f r) (HSMul.hSMul s y))
            -/
            rw [← mul_smul, mul_comm, mul_smul] },
            /-
              🎉 no goals
            -/
        map_add' := fun s₁ s₂ => by
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            f : RingHom R S
            Y : ModuleCat S
            m1 : Module R S := Module.compHom S f
            m2 : Module R ↑Y := Module.compHom (↑Y) f
            s₁ s₂ : S
            ⊢ Eq ((fun s => { toFun := fun y => HSMul.hSMul s y, map_add' := ⋯, map_smul'  …
          -/
          ext y
          /-
            case h
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            f : RingHom R S
            Y : ModuleCat S
            m1 : Module R S := Module.compHom S f
            m2 : Module R ↑Y := Module.compHom (↑Y) f
            s₁ s₂ : S
            y : ↑Y
            ⊢ Eq (((fun s => { toFun := fun y => HSMul.hSMul s y, map_add' := ⋯, map_smul' …
          -/
          change (s₁ + s₂) • y = s₁ • y + s₂ • y
          /-
            case h
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            f : RingHom R S
            Y : ModuleCat S
            m1 : Module R S := Module.compHom S f
            m2 : Module R ↑Y := Module.compHom (↑Y) f
            s₁ s₂ : S
            y : ↑Y
            ⊢ Eq (HSMul.hSMul (HAdd.hAdd s₁ s₂) y) (HAdd.hAdd (HSMul.hSMul s₁ y) (HSMul.hS …
          -/
          rw [add_smul]
          /-
            🎉 no goals
          -/
        map_smul' := fun r s => by
          /-
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            f : RingHom R S
            Y : ModuleCat S
            m1 : Module R S := Module.compHom S f
            m2 : Module R ↑Y := Module.compHom (↑Y) f
            r : R
            s : S
            ⊢ Eq ({ toFun := fun s => { toFun := fun y => HSMul.hSMul s y, map_add' := ⋯,  …
          -/
          ext y
          /-
            case h
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            f : RingHom R S
            Y : ModuleCat S
            m1 : Module R S := Module.compHom S f
            m2 : Module R ↑Y := Module.compHom (↑Y) f
            r : R
            s : S
            y : ↑Y
            ⊢ Eq (({ toFun := fun s => { toFun := fun y => HSMul.hSMul s y, map_add' := ⋯, …
          -/
          change (f r • s) • y = (f r) • s • y
          /-
            case h
            R✝ : Type u₁
            S✝ : Type u₂
            inst✝³ : CommRing R✝
            inst✝² : CommRing S✝
            f✝ : RingHom R✝ S✝
            R : Type u₁
            S : Type u₂
            inst✝¹ : CommRing R
            inst✝ : CommRing S
            f : RingHom R S
            Y : ModuleCat S
            m1 : Module R S := Module.compHom S f
            m2 : Module R ↑Y := Module.compHom (↑Y) f
            r : R
            s : S
            y : ↑Y
            ⊢ Eq (HSMul.hSMul (HSMul.hSMul (f r) s) y) (HSMul.hSMul (f r) (HSMul.hSMul s y))
          -/
          rw [smul_eq_mul, mul_smul] }
          /-
            🎉 no goals
          -/
                              /-
                                R✝ : Type u₁
                                S✝ : Type u₂
                                inst✝³ : CommRing R✝
                                inst✝² : CommRing S✝
                                f✝ : RingHom R✝ S✝
                                R : Type u₁
                                S : Type u₂
                                inst✝¹ : CommRing R
                                inst✝ : CommRing S
                                f : RingHom R S
                                Y : ModuleCat S
                                x✝¹ x✝ : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S …
                                ⊢ Eq ((TensorProduct.lift { toFun := fun s => { toFun := fun y => HSMul.hSMul  …
                              -/
    map_add' := fun _ _ => by rw [map_add]
                              /-
                                🎉 no goals
                              -/
    map_smul' := fun s z => by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        Y : ModuleCat S
        s : S
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑( …
        ⊢ Eq ({ toFun := ⇑(TensorProduct.lift { toFun := fun s => { toFun := fun y =>  …
      -/
      letI m1 : Module R S := Module.compHom S f
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        Y : ModuleCat S
        s : S
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑( …
        m1 : Module R S := Module.compHom S f
        ⊢ Eq ({ toFun := ⇑(TensorProduct.lift { toFun := fun s => { toFun := fun y =>  …
      -/
      letI m2 : Module R Y := Module.compHom Y f
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        Y : ModuleCat S
        s : S
        z : TensorProduct R ↑((ModuleCat.restrictScalars f).obj (ModuleCat.of S S)) ↑( …
        m1 : Module R S := Module.compHom S f
        m2 : Module R ↑Y := Module.compHom (↑Y) f
        ⊢ Eq ({ toFun := ⇑(TensorProduct.lift { toFun := fun s => { toFun := fun y =>  …
      -/
      dsimp only
      induction z using TensorProduct.induction_on with
      | zero => rw [smul_zero, map_zero, smul_zero]
      | tmul s' y =>
        rw [ExtendScalars.smul_tmul, LinearMap.coe_mk]
        erw [TensorProduct.lift.tmul, TensorProduct.lift.tmul]
        set s' : S := s'
        change (s * s') • y = s • s' • y
        rw [mul_smul]
      | add _ _ ih1 ih2 => rw [smul_add, map_add, map_add, ih1, ih2, smul_add] }


/-- The natural transformation from the composition of restriction and extension of scalars to the
identity functor on `S`-module.
-/
@[simps app]
def counit : restrictScalars.{max v u₂,u₁,u₂} f ⋙ extendScalars f ⟶ 𝟭 (ModuleCat S) where
  app _ := Counit.map.{u₁,u₂,v} f
  naturality Y Y' g := by
    -- Porting note: this is very annoying; fix instances in concrete categories
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      Y Y' : ModuleCat S
      g : Quiver.Hom Y Y'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((ModuleCat.restrictScalars f).comp  …
    -/
    letI m1 : Module R S := Module.compHom S f
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      Y Y' : ModuleCat S
      g : Quiver.Hom Y Y'
      m1 : Module R S := Module.compHom S f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((ModuleCat.restrictScalars f).comp  …
    -/
    letI m2 : Module R Y := Module.compHom Y f
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      Y Y' : ModuleCat S
      g : Quiver.Hom Y Y'
      m1 : Module R S := Module.compHom S f
      m2 : Module R ↑Y := Module.compHom (↑Y) f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((ModuleCat.restrictScalars f).comp  …
    -/
    letI m2 : Module R Y' := Module.compHom Y' f
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝³ : CommRing R✝
      inst✝² : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u₁
      S : Type u₂
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      Y Y' : ModuleCat S
      g : Quiver.Hom Y Y'
      m1 : Module R S := Module.compHom S f
      m2✝ : Module R ↑Y := Module.compHom (↑Y) f
      m2 : Module R ↑Y' := Module.compHom (↑Y') f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((ModuleCat.restrictScalars f).comp  …
    -/
    ext z
    induction z using TensorProduct.induction_on with
    | zero => rw [map_zero, map_zero]
    | tmul s' y =>
      dsimp
      -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [Counit.map_hom_apply]
      rw [lift.tmul, LinearMap.coe_mk, LinearMap.coe_mk]
      set s' : S := s'
      change s' • g y = g (s' • y)
      rw [map_smul]
    | add _ _ ih₁ ih₂ => rw [map_add, map_add]; congr 1

/-- Given commutative rings `R, S` and a ring hom `f : R →+* S`, the extension and restriction of
scalars by `f` are adjoint to each other.
-/
-- @[simps] -- Porting note: removed not in normal form and not used
def extendRestrictScalarsAdj {R : Type u₁} {S : Type u₂} [CommRing R] [CommRing S] (f : R →+* S) :
    extendScalars.{u₁,u₂,max v u₂} f ⊣ restrictScalars.{max v u₂,u₁,u₂} f :=
  Adjunction.mk' {
    homEquiv := fun _ _ ↦ ExtendRestrictScalarsAdj.homEquiv.{v,u₁,u₂} f
    unit := ExtendRestrictScalarsAdj.unit.{v,u₁,u₂} f
    counit := ExtendRestrictScalarsAdj.counit.{v,u₁,u₂} f
    homEquiv_unit := fun {X Y g} ↦ hom_ext <| LinearMap.ext fun x => by
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        x : ↑X
        ⊢ Eq ((((fun x x_1 => ModuleCat.ExtendRestrictScalarsAdj.homEquiv f) X Y) g).h …
      -/
      dsimp
      /-
        R✝ : Type u₁
        S✝ : Type u₂
        inst✝³ : CommRing R✝
        inst✝² : CommRing S✝
        f✝ : RingHom R✝ S✝
        R : Type u₁
        S : Type u₂
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        X : ModuleCat R
        Y : ModuleCat S
        g : Quiver.Hom ((ModuleCat.extendScalars f).obj X) Y
        x : ↑X
        ⊢ Eq (((ModuleCat.ExtendRestrictScalarsAdj.homEquiv f) g).hom x) (g.hom ((Modu …
      -/
      rfl
      /-
        🎉 no goals
      -/
    homEquiv_counit := fun {X Y g} ↦ hom_ext <| LinearMap.ext fun x => by
        induction x using TensorProduct.induction_on with
        | zero => rw [map_zero, map_zero]
        | tmul =>
          rw [ExtendRestrictScalarsAdj.homEquiv_symm_apply]
          dsimp
          -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
          erw [ExtendRestrictScalarsAdj.Counit.map_hom_apply,
              ExtendRestrictScalarsAdj.HomEquiv.fromExtendScalars_hom_apply]
        | add => rw [map_add, map_add]; congr 1 }


lemma extendRestrictScalarsAdj_homEquiv_apply
    {R : Type u₁} {S : Type u₂} [CommRing R] [CommRing S]
    {f : R →+* S} {M : ModuleCat.{max v u₂} R} {N : ModuleCat S}
    (φ : (extendScalars f).obj M ⟶ N) (m : M):
    (extendRestrictScalarsAdj f).homEquiv _ _ φ m = φ ((1 : S) ⊗ₜ m) :=
  rfl


lemma extendRestrictScalarsAdj_unit_app_apply
    {R : Type u₁} {S : Type u₂} [CommRing R] [CommRing S]
    (f : R →+* S) (M : ModuleCat.{max v u₂} R) (m : M):
    (extendRestrictScalarsAdj f).unit.app M m = (1 : S) ⊗ₜ[R,f] m :=
  rfl


instance {R : Type u₁} {S : Type u₂} [CommRing R] [CommRing S] (f : R →+* S) :
    (extendScalars.{u₁, u₂, max u₂ w} f).IsLeftAdjoint :=
  (extendRestrictScalarsAdj f).isLeftAdjoint


instance {R : Type u₁} {S : Type u₂} [CommRing R] [CommRing S] (f : R →+* S) :
    (restrictScalars.{max u₂ w, u₁, u₂} f).IsRightAdjoint :=
  (extendRestrictScalarsAdj f).isRightAdjoint


noncomputable instance preservesLimit_restrictScalars
    {R : Type*} {S : Type*} [Ring R] [Ring S] (f : R →+* S) {J : Type*} [Category J]
    (F : J ⥤ ModuleCat.{v} S) [Small.{v} (F ⋙ forget _).sections] :
    PreservesLimit F (restrictScalars f) :=
  ⟨fun {c} hc => ⟨by
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝⁵ : CommRing R✝
      inst✝⁴ : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u_1
      S : Type u_2
      inst✝³ : Ring R
      inst✝² : Ring S
      f : RingHom R S
      J : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} J
      F : CategoryTheory.Functor J (ModuleCat S)
      inst✝ : Small.{v, max u_3 v} ↑(F.comp (CategoryTheory.forget (ModuleCat S))).s …
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit ((ModuleCat.restrictScalars f).mapCone c)
    -/
    have hc' := isLimitOfPreserves (forget₂ _ AddCommGrp) hc
    /-
      R✝ : Type u₁
      S✝ : Type u₂
      inst✝⁵ : CommRing R✝
      inst✝⁴ : CommRing S✝
      f✝ : RingHom R✝ S✝
      R : Type u_1
      S : Type u_2
      inst✝³ : Ring R
      inst✝² : Ring S
      f : RingHom R S
      J : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} J
      F : CategoryTheory.Functor J (ModuleCat S)
      inst✝ : Small.{v, max u_3 v} ↑(F.comp (CategoryTheory.forget (ModuleCat S))).s …
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hc' : CategoryTheory.Limits.IsLimit ((CategoryTheory.forget₂ (ModuleCat S) Add …
      ⊢ CategoryTheory.Limits.IsLimit ((ModuleCat.restrictScalars f).mapCone c)
    -/
    exact isLimitOfReflects (forget₂ _ AddCommGrp) hc'⟩⟩
    /-
      🎉 no goals
    -/


instance preservesColimit_restrictScalars {R S : Type*} [Ring R] [Ring S]
    (f : R →+* S) {J : Type*} [Category J] (F : J ⥤ ModuleCat.{v} S)
    [HasColimit (F ⋙ forget₂ _ AddCommGrp)] :
    PreservesColimit F (ModuleCat.restrictScalars.{v} f) := by
  have : HasColimit ((F ⋙ restrictScalars f) ⋙ forget₂ (ModuleCat R) AddCommGrp) :=
    inferInstanceAs (HasColimit (F ⋙ forget₂ _ AddCommGrp))
  /-
    R✝ : Type u₁
    S✝ : Type u₂
    inst✝⁵ : CommRing R✝
    inst✝⁴ : CommRing S✝
    f✝ : RingHom R✝ S✝
    R : Type u_1
    S : Type u_2
    inst✝³ : Ring R
    inst✝² : Ring S
    f : RingHom R S
    J : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} J
    F : CategoryTheory.Functor J (ModuleCat S)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
    this : CategoryTheory.Limits.HasColimit ((F.comp (ModuleCat.restrictScalars f) …
    ⊢ CategoryTheory.Limits.PreservesColimit F (ModuleCat.restrictScalars f)
  -/
  apply preservesColimit_of_preserves_colimit_cocone (HasColimit.isColimitColimitCocone F)
  /-
    R✝ : Type u₁
    S✝ : Type u₂
    inst✝⁵ : CommRing R✝
    inst✝⁴ : CommRing S✝
    f✝ : RingHom R✝ S✝
    R : Type u_1
    S : Type u_2
    inst✝³ : Ring R
    inst✝² : Ring S
    f : RingHom R S
    J : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} J
    F : CategoryTheory.Functor J (ModuleCat S)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
    this : CategoryTheory.Limits.HasColimit ((F.comp (ModuleCat.restrictScalars f) …
    ⊢ CategoryTheory.Limits.IsColimit ((ModuleCat.restrictScalars f).mapCocone (Mo …
  -/
  apply isColimitOfReflects (forget₂ _ AddCommGrp)
  /-
    case t
    R✝ : Type u₁
    S✝ : Type u₂
    inst✝⁵ : CommRing R✝
    inst✝⁴ : CommRing S✝
    f✝ : RingHom R✝ S✝
    R : Type u_1
    S : Type u_2
    inst✝³ : Ring R
    inst✝² : Ring S
    f : RingHom R S
    J : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} J
    F : CategoryTheory.Functor J (ModuleCat S)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
    this : CategoryTheory.Limits.HasColimit ((F.comp (ModuleCat.restrictScalars f) …
    ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.forget₂ (ModuleCat R) AddCo …
  -/
  apply isColimitOfPreserves (forget₂ (ModuleCat.{v} S) AddCommGrp.{v})
  /-
    case t.t
    R✝ : Type u₁
    S✝ : Type u₂
    inst✝⁵ : CommRing R✝
    inst✝⁴ : CommRing S✝
    f✝ : RingHom R✝ S✝
    R : Type u_1
    S : Type u_2
    inst✝³ : Ring R
    inst✝² : Ring S
    f : RingHom R S
    J : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} J
    F : CategoryTheory.Functor J (ModuleCat S)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.forget₂ (Modu …
    this : CategoryTheory.Limits.HasColimit ((F.comp (ModuleCat.restrictScalars f) …
    ⊢ CategoryTheory.Limits.IsColimit (ModuleCat.HasColimit.colimitCocone F)
  -/
  exact HasColimit.isColimitColimitCocone F
  /-
    🎉 no goals
  -/


variable (R) in
/-- The extension of scalars by the identity of a ring is isomorphic to the
identity functor. -/
noncomputable def extendScalarsId : extendScalars (RingHom.id R) ≅ 𝟭 _ :=
  ((conjugateIsoEquiv (extendRestrictScalarsAdj (RingHom.id R)) Adjunction.id).symm
    (restrictScalarsId R)).symm


lemma extendScalarsId_inv_app_apply (M : ModuleCat R) (m : M):
    (extendScalarsId R).inv.app M m = (1 : R) ⊗ₜ m := rfl


lemma homEquiv_extendScalarsId (M : ModuleCat R) :
    (extendRestrictScalarsAdj (RingHom.id R)).homEquiv _ _ ((extendScalarsId R).hom.app M) =
      (restrictScalarsId R).inv.app M := by
  /-
    R : Type u₁
    inst✝ : CommRing R
    M : ModuleCat R
    ⊢ Eq (((ModuleCat.extendRestrictScalarsAdj (RingHom.id R)).homEquiv M ((Catego …
  -/
  ext m
  /-
    case hf.h
    R : Type u₁
    inst✝ : CommRing R
    M : ModuleCat R
    m : ↑M
    ⊢ Eq ((((ModuleCat.extendRestrictScalarsAdj (RingHom.id R)).homEquiv M ((Categ …
  -/
  rw [extendRestrictScalarsAdj_homEquiv_apply, ← extendScalarsId_inv_app_apply]
  /-
    case hf.h
    R : Type u₁
    inst✝ : CommRing R
    M : ModuleCat R
    m : ↑M
    ⊢ Eq (((ModuleCat.extendScalarsId R).hom.app M).hom (((ModuleCat.extendScalars …
  -/
  erw [← comp_apply]
  /-
    case hf.h
    R : Type u₁
    inst✝ : CommRing R
    M : ModuleCat R
    m : ↑M
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((ModuleCat.extendScalarsId R).inv.a …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma extendScalarsId_hom_app_one_tmul (M : ModuleCat R) (m : M) :
    (extendScalarsId R).hom.app M ((1 : R) ⊗ₜ m) = m := by
  rw [← extendRestrictScalarsAdj_homEquiv_apply,
    homEquiv_extendScalarsId]
  /-
    R : Type u₁
    inst✝ : CommRing R
    M : ModuleCat R
    m : ↑M
    ⊢ Eq (((ModuleCat.restrictScalarsId R).inv.app M).hom m) m
  -/
  dsimp
  /-
    🎉 no goals
  -/


/-- The extension of scalars by a composition of commutative ring morphisms
identify to the composition of the extension of scalars functors. -/
noncomputable def extendScalarsComp :
    extendScalars (f₂₃.comp f₁₂) ≅ extendScalars f₁₂ ⋙ extendScalars f₂₃ :=
  (conjugateIsoEquiv
    ((extendRestrictScalarsAdj f₁₂).comp (extendRestrictScalarsAdj f₂₃))
    (extendRestrictScalarsAdj (f₂₃.comp f₁₂))).symm (restrictScalarsComp f₁₂ f₂₃).symm


lemma homEquiv_extendScalarsComp (M : ModuleCat R₁) :
    (extendRestrictScalarsAdj (f₂₃.comp f₁₂)).homEquiv _ _
      ((extendScalarsComp f₁₂ f₂₃).hom.app M) =
      (extendRestrictScalarsAdj f₁₂).unit.app M ≫
        (restrictScalars f₁₂).map ((extendRestrictScalarsAdj f₂₃).unit.app _) ≫
        (restrictScalarsComp f₁₂ f₂₃).inv.app _ := by
  /-
    R₁ R₂ R₃ : Type u₁
    inst✝² : CommRing R₁
    inst✝¹ : CommRing R₂
    inst✝ : CommRing R₃
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    M : ModuleCat R₁
    ⊢ Eq (((ModuleCat.extendRestrictScalarsAdj (f₂₃.comp f₁₂)).homEquiv M (((Modul …
  -/
  dsimp [extendScalarsComp, conjugateIsoEquiv, conjugateEquiv]
  simp only [Category.assoc, Category.id_comp, Category.comp_id,
    Adjunction.comp_unit_app, Adjunction.homEquiv_unit,
    Functor.map_comp, Adjunction.unit_naturality_assoc,
    Adjunction.right_triangle_components]
  /-
    R₁ R₂ R₃ : Type u₁
    inst✝² : CommRing R₁
    inst✝¹ : CommRing R₂
    inst✝ : CommRing R₃
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    M : ModuleCat R₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.extendRestrictScalarsAdj  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma extendScalarsComp_hom_app_one_tmul (M : ModuleCat R₁) (m : M) :
    (extendScalarsComp f₁₂ f₂₃).hom.app M ((1 : R₃) ⊗ₜ m) =
      (1 : R₃) ⊗ₜ[R₂,f₂₃] ((1 : R₂) ⊗ₜ[R₁,f₁₂] m) := by
  /-
    R₁ R₂ R₃ : Type u₁
    inst✝² : CommRing R₁
    inst✝¹ : CommRing R₂
    inst✝ : CommRing R₃
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (((ModuleCat.extendScalarsComp f₁₂ f₂₃).hom.app M).hom (TensorProduct.tmu …
  -/
  rw [← extendRestrictScalarsAdj_homEquiv_apply, homEquiv_extendScalarsComp]
  /-
    R₁ R₂ R₃ : Type u₁
    inst✝² : CommRing R₁
    inst✝¹ : CommRing R₂
    inst✝ : CommRing R₃
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((ModuleCat.extendRestrictScalarsAdj …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma extendScalars_assoc :
    (extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom ≫ whiskerRight (extendScalarsComp f₁₂ f₂₃).hom _ =
      (extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom ≫ whiskerLeft _ (extendScalarsComp f₂₃ f₃₄).hom ≫
        (Functor.associator _ _ _).inv := by
  /-
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.com …
  -/
  ext M m
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.c …
  -/
  have h₁ := extendScalarsComp_hom_app_one_tmul (f₂₃.comp f₁₂) f₃₄ M m
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.c …
  -/
  have h₂ := extendScalarsComp_hom_app_one_tmul f₁₂ (f₃₄.comp f₂₃) M m
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    h₂ : Eq (((ModuleCat.extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom.app M).hom (Ten …
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.c …
  -/
  have h₃ := extendScalarsComp_hom_app_one_tmul f₂₃ f₃₄
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    h₂ : Eq (((ModuleCat.extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom.app M).hom (Ten …
    h₃ : ∀ (M : ModuleCat R₂) (m : ↑M), Eq (((ModuleCat.extendScalarsComp f₂₃ f₃₄) …
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.c …
  -/
  have h₄ := extendScalarsComp_hom_app_one_tmul f₁₂ f₂₃ M m
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    h₂ : Eq (((ModuleCat.extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom.app M).hom (Ten …
    h₃ : ∀ (M : ModuleCat R₂) (m : ↑M), Eq (((ModuleCat.extendScalarsComp f₂₃ f₃₄) …
    h₄ : Eq (((ModuleCat.extendScalarsComp f₁₂ f₂₃).hom.app M).hom (TensorProduct. …
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.c …
  -/
  dsimp at h₁ h₂ h₃ h₄ ⊢
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    h₂ : Eq (((ModuleCat.extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom.app M).hom (Ten …
    h₃ : ∀ (M : ModuleCat R₂) (m : ↑M), Eq (((ModuleCat.extendScalarsComp f₂₃ f₃₄) …
    h₄ : Eq (((ModuleCat.extendScalarsComp f₁₂ f₂₃).hom.app M).hom (TensorProduct. …
    ⊢ Eq (((ModuleCat.extendScalars f₃₄).map ((ModuleCat.extendScalarsComp f₁₂ f₂₃ …
  -/
  rw [h₁]
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    h₂ : Eq (((ModuleCat.extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom.app M).hom (Ten …
    h₃ : ∀ (M : ModuleCat R₂) (m : ↑M), Eq (((ModuleCat.extendScalarsComp f₂₃ f₃₄) …
    h₄ : Eq (((ModuleCat.extendScalarsComp f₁₂ f₂₃).hom.app M).hom (TensorProduct. …
    ⊢ Eq (((ModuleCat.extendScalars f₃₄).map ((ModuleCat.extendScalarsComp f₁₂ f₂₃ …
  -/
  erw [h₂]
  /-
    case w.h.h
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    M : ModuleCat R₁
    m : ↑M
    h₁ : Eq (((ModuleCat.extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom.app M).hom (Ten …
    h₂ : Eq (((ModuleCat.extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).hom.app M).hom (Ten …
    h₃ : ∀ (M : ModuleCat R₂) (m : ↑M), Eq (((ModuleCat.extendScalarsComp f₂₃ f₃₄) …
    h₄ : Eq (((ModuleCat.extendScalarsComp f₁₂ f₂₃).hom.app M).hom (TensorProduct. …
    ⊢ Eq (((ModuleCat.extendScalars f₃₄).map ((ModuleCat.extendScalarsComp f₁₂ f₂₃ …
  -/
  rw [h₃, ExtendScalars.map_tmul, h₄]
  /-
    🎉 no goals
  -/


/-- The associativity compatibility for the extension of scalars, in the exact form
that is needed in the definition `CommRingCat.moduleCatExtendScalarsPseudofunctor`
in the file `Algebra.Category.ModuleCat.Pseudofunctor` -/
lemma extendScalars_assoc' :
    (extendScalarsComp (f₂₃.comp f₁₂) f₃₄).hom ≫ whiskerRight (extendScalarsComp f₁₂ f₂₃).hom _ ≫
      (Functor.associator _ _ _).hom ≫ whiskerLeft _ (extendScalarsComp f₂₃ f₃₄).inv ≫
        (extendScalarsComp f₁₂ (f₃₄.comp f₂₃)).inv = 𝟙 _ := by
  /-
    R₁ R₂ R₃ R₄ : Type u₁
    inst✝³ : CommRing R₁
    inst✝² : CommRing R₂
    inst✝¹ : CommRing R₃
    inst✝ : CommRing R₄
    f₁₂ : RingHom R₁ R₂
    f₂₃ : RingHom R₂ R₃
    f₃₄ : RingHom R₃ R₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (f₂₃.com …
  -/
  rw [extendScalars_assoc_assoc]
  simp only [Iso.inv_hom_id_assoc, ← whiskerLeft_comp_assoc, Iso.hom_inv_id,
    whiskerLeft_id', Category.id_comp]


@[reassoc]
lemma extendScalars_id_comp :
    (extendScalarsComp (RingHom.id R₁) f₁₂).hom ≫ whiskerRight (extendScalarsId R₁).hom _ ≫
      (Functor.leftUnitor _).hom = 𝟙 _ := by
  /-
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (RingHom …
  -/
  ext M m
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp (RingH …
  -/
  dsimp
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (((ModuleCat.extendScalars f₁₂).map ((ModuleCat.extendScalarsId R₁).hom.a …
  -/
  erw [extendScalarsComp_hom_app_one_tmul (RingHom.id R₁) f₁₂ M m]
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (((ModuleCat.extendScalars f₁₂).map ((ModuleCat.extendScalarsId R₁).hom.a …
  -/
  rw [ExtendScalars.map_tmul]
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (TensorProduct.tmul R₁ 1 (((ModuleCat.extendScalarsId R₁).hom.app M).hom  …
  -/
  erw [extendScalarsId_hom_app_one_tmul]
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (TensorProduct.tmul R₁ 1 m) (LinearMap.id (TensorProduct.tmul R₁ 1 m))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma extendScalars_comp_id :
    (extendScalarsComp f₁₂ (RingHom.id R₂)).hom ≫ whiskerLeft _ (extendScalarsId R₂).hom ≫
      (Functor.rightUnitor _).hom = 𝟙 _ := by
  /-
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp f₁₂ (Rin …
  -/
  ext M m
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (((CategoryTheory.CategoryStruct.comp (ModuleCat.extendScalarsComp f₁₂ (R …
  -/
  dsimp
  erw [extendScalarsComp_hom_app_one_tmul f₁₂ (RingHom.id R₂) M m,
    extendScalarsId_hom_app_one_tmul]
  /-
    case w.h.h
    R₁ R₂ : Type u₁
    inst✝¹ : CommRing R₁
    inst✝ : CommRing R₂
    f₁₂ : RingHom R₁ R₂
    M : ModuleCat R₁
    m : ↑M
    ⊢ Eq (TensorProduct.tmul R₁ 1 m) (LinearMap.id (TensorProduct.tmul R₁ 1 m))
  -/
  rfl
  /-
    🎉 no goals
  -/


