/-- The type of derivations with values in a `B`-module `M` relative
to a morphism `f : A ⟶ B` in the category `CommRingCat`. -/
nonrec def Derivation : Type _ :=
  letI := f.hom.toAlgebra
  letI := Module.compHom M f.hom
  Derivation A B M


/-- Constructor for `ModuleCat.Derivation`. -/
def mk (d : B → M) (d_add : ∀ (b b' : B), d (b + b') = d b + d b' := by simp)
    (d_mul : ∀ (b b' : B), d (b * b') = b • d b' + b' • d b := by simp)
    (d_map : ∀ (a : A), d (f a) = 0 := by simp) :
    M.Derivation f :=
  letI := f.hom.toAlgebra
  letI := Module.compHom M f.hom
  { toFun := d
    map_add' := d_add
    map_smul' := fun a b ↦ by
      /-
        A B : CommRingCat
        M : ModuleCat ↑B
        f : Quiver.Hom A B
        d : ↑B → ↑M
        d_add : autoParam (∀ (b b' : ↑B), Eq (d (HAdd.hAdd b b')) (HAdd.hAdd (d b) (d  …
        d_mul : autoParam (∀ (b b' : ↑B), Eq (d (HMul.hMul b b')) (HAdd.hAdd (HSMul.hS …
        d_map : autoParam (∀ (a : ↑A), Eq (d (f.hom a)) 0) _auto✝
        this✝ : Algebra ↑A ↑B := f.hom.toAlgebra
        this : Module ↑A ↑M := Module.compHom (↑M) f.hom
        a : ↑A
        b : ↑B
        ⊢ Eq ({ toFun := d, map_add' := d_add }.toFun (HSMul.hSMul a b)) (HSMul.hSMul  …
      -/
      dsimp
      /-
        A B : CommRingCat
        M : ModuleCat ↑B
        f : Quiver.Hom A B
        d : ↑B → ↑M
        d_add : autoParam (∀ (b b' : ↑B), Eq (d (HAdd.hAdd b b')) (HAdd.hAdd (d b) (d  …
        d_mul : autoParam (∀ (b b' : ↑B), Eq (d (HMul.hMul b b')) (HAdd.hAdd (HSMul.hS …
        d_map : autoParam (∀ (a : ↑A), Eq (d (f.hom a)) 0) _auto✝
        this✝ : Algebra ↑A ↑B := f.hom.toAlgebra
        this : Module ↑A ↑M := Module.compHom (↑M) f.hom
        a : ↑A
        b : ↑B
        ⊢ Eq (d (HSMul.hSMul a b)) (HSMul.hSMul a (d b))
      -/
      erw [d_mul, d_map, smul_zero, add_zero]
      /-
        A B : CommRingCat
        M : ModuleCat ↑B
        f : Quiver.Hom A B
        d : ↑B → ↑M
        d_add : autoParam (∀ (b b' : ↑B), Eq (d (HAdd.hAdd b b')) (HAdd.hAdd (d b) (d  …
        d_mul : autoParam (∀ (b b' : ↑B), Eq (d (HMul.hMul b b')) (HAdd.hAdd (HSMul.hS …
        d_map : autoParam (∀ (a : ↑A), Eq (d (f.hom a)) 0) _auto✝
        this✝ : Algebra ↑A ↑B := f.hom.toAlgebra
        this : Module ↑A ↑M := Module.compHom (↑M) f.hom
        a : ↑A
        b : ↑B
        ⊢ Eq (HSMul.hSMul (f.hom a) (d b)) (HSMul.hSMul a (d b))
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_one_eq_zero' := by
      /-
        A B : CommRingCat
        M : ModuleCat ↑B
        f : Quiver.Hom A B
        d : ↑B → ↑M
        d_add : autoParam (∀ (b b' : ↑B), Eq (d (HAdd.hAdd b b')) (HAdd.hAdd (d b) (d  …
        d_mul : autoParam (∀ (b b' : ↑B), Eq (d (HMul.hMul b b')) (HAdd.hAdd (HSMul.hS …
        d_map : autoParam (∀ (a : ↑A), Eq (d (f.hom a)) 0) _auto✝
        this✝ : Algebra ↑A ↑B := f.hom.toAlgebra
        this : Module ↑A ↑M := Module.compHom (↑M) f.hom
        ⊢ Eq ({ toFun := d, map_add' := d_add, map_smul' := ⋯ } 1) 0
      -/
      dsimp
      /-
        A B : CommRingCat
        M : ModuleCat ↑B
        f : Quiver.Hom A B
        d : ↑B → ↑M
        d_add : autoParam (∀ (b b' : ↑B), Eq (d (HAdd.hAdd b b')) (HAdd.hAdd (d b) (d  …
        d_mul : autoParam (∀ (b b' : ↑B), Eq (d (HMul.hMul b b')) (HAdd.hAdd (HSMul.hS …
        d_map : autoParam (∀ (a : ↑A), Eq (d (f.hom a)) 0) _auto✝
        this✝ : Algebra ↑A ↑B := f.hom.toAlgebra
        this : Module ↑A ↑M := Module.compHom (↑M) f.hom
        ⊢ Eq (d 1) 0
      -/
      rw [← f.hom.map_one, d_map]
      /-
        🎉 no goals
      -/
    leibniz' := d_mul }


/-- The underlying map `B → M` of a derivation `M.Derivation f` when `M : ModuleCat B`
and `f : A ⟶ B` is a morphism in `CommRingCat`. -/
def d (b : B) : M :=
  letI := f.hom.toAlgebra
  letI := Module.compHom M f.hom
  _root_.Derivation.toLinearMap D b


@[simp]
                                                             /-
                                                               A B : CommRingCat
                                                               M : ModuleCat ↑B
                                                               f : Quiver.Hom A B
                                                               D : M.Derivation f
                                                               b b' : ↑B
                                                               ⊢ Eq (D.d (HAdd.hAdd b b')) (HAdd.hAdd (D.d b) (D.d b'))
                                                             -/
lemma d_add (b b' : B) : D.d (b + b') = D.d b + D.d b' := by simp [d]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                                      /-
                                                                        A B : CommRingCat
                                                                        M : ModuleCat ↑B
                                                                        f : Quiver.Hom A B
                                                                        D : M.Derivation f
                                                                        b b' : ↑B
                                                                        ⊢ Eq (D.d (HMul.hMul b b')) (HAdd.hAdd (HSMul.hSMul b (D.d b')) (HSMul.hSMul b …
                                                                      -/
lemma d_mul (b b' : B) : D.d (b * b') = b • D.d b' + b' • D.d b := by simp [d]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
lemma d_map (a : A) : D.d (f a) = 0 :=
  letI := f.hom.toAlgebra
  letI := Module.compHom M f.hom
  D.map_algebraMap a


variable (f) in
/-- The module of differentials of a morphism `f : A ⟶ B` in the category `CommRingCat`. -/
noncomputable def KaehlerDifferential : ModuleCat.{u} B :=
  letI := f.hom.toAlgebra
  ModuleCat.of B (_root_.KaehlerDifferential A B)


variable (f) in
/-- The (universal) derivation in `(KaehlerDifferential f).Derivation f`
when `f : A ⟶ B` is a morphism in the category `CommRingCat`. -/
noncomputable def D : (KaehlerDifferential f).Derivation f :=
  letI := f.hom.toAlgebra
  ModuleCat.Derivation.mk
                                                     /-
                                                       A B A' B' : CommRingCat
                                                       f : Quiver.Hom A B
                                                       f' : Quiver.Hom A' B'
                                                       g : Quiver.Hom A A'
                                                       g' : Quiver.Hom B B'
                                                       fac : Eq (CategoryTheory.CategoryStruct.comp g f') (CategoryTheory.CategoryStr …
                                                       this : Algebra ↑A ↑B := f.hom.toAlgebra
                                                       ⊢ ∀ (b b' : ↑B), Eq ((fun b => (_root_.KaehlerDifferential.D ↑A ↑B) b) (HAdd.h …
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    (fun b ↦ _root_.KaehlerDifferential.D A B b) (by simp) (by simp)
                                                               /-
                                                                 🎉 no goals
                                                               -/
      (_root_.KaehlerDifferential.D A B).map_algebraMap


/-- When `f : A ⟶ B` is a morphism in the category `CommRingCat`, this is the
differential map `B → KaehlerDifferential f`. -/
noncomputable abbrev d (b : B) : KaehlerDifferential f := (D f).d b


@[ext]
lemma ext {M : ModuleCat B} {α β : KaehlerDifferential f ⟶ M}
    (h : ∀ (b : B), α (d b) = β (d b)) : α = β := by
  /-
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    α β : Quiver.Hom (CommRingCat.KaehlerDifferential f) M
    h : ∀ (b : ↑B), Eq (α.hom (CommRingCat.KaehlerDifferential.d b)) (β.hom (CommR …
    ⊢ Eq α β
  -/
  rw [← sub_eq_zero]
  have : ⊤ ≤ LinearMap.ker (α - β).hom := by
    rw [← KaehlerDifferential.span_range_derivation, Submodule.span_le]
    rintro _ ⟨y, rfl⟩
    rw [SetLike.mem_coe, LinearMap.mem_ker, ModuleCat.hom_sub, LinearMap.sub_apply, sub_eq_zero]
    apply h
  /-
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    α β : Quiver.Hom (CommRingCat.KaehlerDifferential f) M
    h : ∀ (b : ↑B), Eq (α.hom (CommRingCat.KaehlerDifferential.d b)) (β.hom (CommR …
    this : LE.le Top.top (LinearMap.ker (HSub.hSub α β).hom)
    ⊢ Eq (HSub.hSub α β) 0
  -/
  rw [top_le_iff, LinearMap.ker_eq_top] at this
  /-
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    α β : Quiver.Hom (CommRingCat.KaehlerDifferential f) M
    h : ∀ (b : ↑B), Eq (α.hom (CommRingCat.KaehlerDifferential.d b)) (β.hom (CommR …
    this : Eq (HSub.hSub α β).hom 0
    ⊢ Eq (HSub.hSub α β) 0
  -/
  ext : 1
  /-
    case hf
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    α β : Quiver.Hom (CommRingCat.KaehlerDifferential f) M
    h : ∀ (b : ↑B), Eq (α.hom (CommRingCat.KaehlerDifferential.d b)) (β.hom (CommR …
    this : Eq (HSub.hSub α β).hom 0
    ⊢ Eq (HSub.hSub α β).hom (ModuleCat.Hom.hom 0)
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- The map `KaehlerDifferential f ⟶ (ModuleCat.restrictScalars g').obj (KaehlerDifferential f')`
induced by a commutative square (given by an equality `g ≫ f' = f ≫ g'`)
in the category `CommRingCat`. -/
noncomputable def map :
    KaehlerDifferential f ⟶
      (ModuleCat.restrictScalars g'.hom).obj (KaehlerDifferential f') :=
  letI := f.hom.toAlgebra
  letI := f'.hom.toAlgebra
  letI := g.hom.toAlgebra
  letI := g'.hom.toAlgebra
  letI := (g ≫ f').hom.toAlgebra
  have : IsScalarTower A A' B' := IsScalarTower.of_algebraMap_eq' rfl
  have := IsScalarTower.of_algebraMap_eq' (congrArg Hom.hom fac)
  -- TODO: after https://github.com/leanprover-community/mathlib4/pull/19511 we need to hint `(Y := ...)`.
  -- This suggests `restrictScalars` needs to be redesigned.
  ModuleCat.ofHom (Y := (ModuleCat.restrictScalars g'.hom).obj (KaehlerDifferential f'))
  { toFun := fun x ↦ _root_.KaehlerDifferential.map A A' B B' x
                   /-
                     A B A' B' : CommRingCat
                     f : Quiver.Hom A B
                     f' : Quiver.Hom A' B'
                     g : Quiver.Hom A A'
                     g' : Quiver.Hom B B'
                     fac : Eq (CategoryTheory.CategoryStruct.comp g f') (CategoryTheory.CategoryStr …
                     this✝⁵ : Algebra ↑A ↑B := f.hom.toAlgebra
                     this✝⁴ : Algebra ↑A' ↑B' := f'.hom.toAlgebra
                     this✝³ : Algebra ↑A ↑A' := g.hom.toAlgebra
                     this✝² : Algebra ↑B ↑B' := g'.hom.toAlgebra
                     this✝¹ : Algebra ↑A ↑B' := (CategoryTheory.CategoryStruct.comp g f').hom.toAlg …
                     this✝ : IsScalarTower ↑A ↑A' ↑B'
                     this : IsScalarTower ↑A ↑B ↑B'
                     ⊢ ∀ (x y : _root_.KaehlerDifferential ↑A ↑B), Eq ((fun x => (_root_.KaehlerDif …
                   -/
    map_add' := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      A B A' B' : CommRingCat
                      f : Quiver.Hom A B
                      f' : Quiver.Hom A' B'
                      g : Quiver.Hom A A'
                      g' : Quiver.Hom B B'
                      fac : Eq (CategoryTheory.CategoryStruct.comp g f') (CategoryTheory.CategoryStr …
                      this✝⁵ : Algebra ↑A ↑B := f.hom.toAlgebra
                      this✝⁴ : Algebra ↑A' ↑B' := f'.hom.toAlgebra
                      this✝³ : Algebra ↑A ↑A' := g.hom.toAlgebra
                      this✝² : Algebra ↑B ↑B' := g'.hom.toAlgebra
                      this✝¹ : Algebra ↑A ↑B' := (CategoryTheory.CategoryStruct.comp g f').hom.toAlg …
                      this✝ : IsScalarTower ↑A ↑A' ↑B'
                      this : IsScalarTower ↑A ↑B ↑B'
                      ⊢ ∀ (m : ↑B) (x : _root_.KaehlerDifferential ↑A ↑B), Eq ({ toFun := fun x => ( …
                    -/
    map_smul' := by simp }
                    /-
                      🎉 no goals
                    -/


@[simp]
lemma map_d (b : B) : map fac (d b) = d (g' b) := by
  /-
    A B A' B' : CommRingCat
    f : Quiver.Hom A B
    f' : Quiver.Hom A' B'
    g : Quiver.Hom A A'
    g' : Quiver.Hom B B'
    fac : Eq (CategoryTheory.CategoryStruct.comp g f') (CategoryTheory.CategoryStr …
    b : ↑B
    ⊢ Eq ((CommRingCat.KaehlerDifferential.map fac).hom (CommRingCat.KaehlerDiffer …
  -/
  algebraize [f.hom, f'.hom, g.hom, g'.hom, f'.hom.comp g.hom]
  /-
    A B A' B' : CommRingCat
    f : Quiver.Hom A B
    f' : Quiver.Hom A' B'
    g : Quiver.Hom A A'
    g' : Quiver.Hom B B'
    fac : Eq (CategoryTheory.CategoryStruct.comp g f') (CategoryTheory.CategoryStr …
    b : ↑B
    algInst✝⁴ : Algebra ↑A ↑B := f.hom.toAlgebra
    algInst✝³ : Algebra ↑A' ↑B' := f'.hom.toAlgebra
    algInst✝² : Algebra ↑A ↑A' := g.hom.toAlgebra
    algInst✝¹ : Algebra ↑B ↑B' := g'.hom.toAlgebra
    algInst✝ : Algebra ↑A ↑B' := (f'.hom.comp g.hom).toAlgebra
    ⊢ Eq ((CommRingCat.KaehlerDifferential.map fac).hom (CommRingCat.KaehlerDiffer …
  -/
  have := IsScalarTower.of_algebraMap_eq' (congrArg Hom.hom fac)
  /-
    A B A' B' : CommRingCat
    f : Quiver.Hom A B
    f' : Quiver.Hom A' B'
    g : Quiver.Hom A A'
    g' : Quiver.Hom B B'
    fac : Eq (CategoryTheory.CategoryStruct.comp g f') (CategoryTheory.CategoryStr …
    b : ↑B
    algInst✝⁴ : Algebra ↑A ↑B := f.hom.toAlgebra
    algInst✝³ : Algebra ↑A' ↑B' := f'.hom.toAlgebra
    algInst✝² : Algebra ↑A ↑A' := g.hom.toAlgebra
    algInst✝¹ : Algebra ↑B ↑B' := g'.hom.toAlgebra
    algInst✝ : Algebra ↑A ↑B' := (f'.hom.comp g.hom).toAlgebra
    this : IsScalarTower ↑A ↑B ↑B'
    ⊢ Eq ((CommRingCat.KaehlerDifferential.map fac).hom (CommRingCat.KaehlerDiffer …
  -/
  exact _root_.KaehlerDifferential.map_D A A' B B' b
  /-
    🎉 no goals
  -/


/-- Given `f : A ⟶ B` a morphism in the category `CommRingCat`, `M : ModuleCat B`,
and `D : M.Derivation f`, this is the induced
morphism `CommRingCat.KaehlerDifferential f ⟶ M`. -/
noncomputable def desc : CommRingCat.KaehlerDifferential f ⟶ M :=
  letI := f.hom.toAlgebra
  letI := Module.compHom M f.hom
  ofHom D.liftKaehlerDifferential


@[simp]
lemma desc_d (b : B) : D.desc (CommRingCat.KaehlerDifferential.d b) = D.d b := by
  /-
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    D : M.Derivation f
    b : ↑B
    ⊢ Eq (D.desc.hom (CommRingCat.KaehlerDifferential.d b)) (D.d b)
  -/
  letI := f.hom.toAlgebra
  /-
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    D : M.Derivation f
    b : ↑B
    this : Algebra ↑A ↑B := f.hom.toAlgebra
    ⊢ Eq (D.desc.hom (CommRingCat.KaehlerDifferential.d b)) (D.d b)
  -/
  letI := Module.compHom M f.hom
  /-
    A B : CommRingCat
    f : Quiver.Hom A B
    M : ModuleCat ↑B
    D : M.Derivation f
    b : ↑B
    this✝ : Algebra ↑A ↑B := f.hom.toAlgebra
    this : Module ↑A ↑M := Module.compHom (↑M) f.hom
    ⊢ Eq (D.desc.hom (CommRingCat.KaehlerDifferential.d b)) (D.d b)
  -/
  apply D.liftKaehlerDifferential_comp_D
  /-
    🎉 no goals
  -/


