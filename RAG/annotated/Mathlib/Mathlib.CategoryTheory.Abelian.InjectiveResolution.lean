/-- Auxiliary construction for `desc`. -/
def descFZero {Y Z : C} (f : Z ⟶ Y) (I : InjectiveResolution Y) (J : InjectiveResolution Z) :
    J.cocomplex.X 0 ⟶ I.cocomplex.X 0 :=
  factorThru (f ≫ I.ι.f 0) (J.ι.f 0)


lemma exact₀ {Z : C} (I : InjectiveResolution Z) :
    (ShortComplex.mk _ _ I.ι_f_zero_comp_complex_d).Exact :=
  ShortComplex.exact_of_f_is_kernel _ I.isLimitKernelFork


/-- Auxiliary construction for `desc`. -/
def descFOne {Y Z : C} (f : Z ⟶ Y) (I : InjectiveResolution Y) (J : InjectiveResolution Z) :
    J.cocomplex.X 1 ⟶ I.cocomplex.X 1 :=
  J.exact₀.descToInjective (descFZero f I J ≫ I.cocomplex.d 0 1)
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Abelian C
          Y Z : C
          f : Quiver.Hom Z Y
          I : CategoryTheory.InjectiveResolution Y
          J : CategoryTheory.InjectiveResolution Z
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.mk (J.ι. …
        -/
    (by dsimp; simp only [← assoc, descFZero]; simp [assoc])
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem descFOne_zero_comm {Y Z : C} (f : Z ⟶ Y) (I : InjectiveResolution Y)
    (J : InjectiveResolution Z) :
    J.cocomplex.d 0 1 ≫ descFOne f I J = descFZero f I J ≫ I.cocomplex.d 0 1 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    Y Z : C
    f : Quiver.Hom Z Y
    I : CategoryTheory.InjectiveResolution Y
    J : CategoryTheory.InjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.cocomplex.d 0 1) (CategoryTheory.I …
  -/
  apply J.exact₀.comp_descToInjective
  /-
    🎉 no goals
  -/


/-- Auxiliary construction for `desc`. -/
def descFSucc {Y Z : C} (I : InjectiveResolution Y) (J : InjectiveResolution Z) (n : ℕ)
    (g : J.cocomplex.X n ⟶ I.cocomplex.X n) (g' : J.cocomplex.X (n + 1) ⟶ I.cocomplex.X (n + 1))
    (w : J.cocomplex.d n (n + 1) ≫ g' = g ≫ I.cocomplex.d n (n + 1)) :
    Σ'g'' : J.cocomplex.X (n + 2) ⟶ I.cocomplex.X (n + 2),
      J.cocomplex.d (n + 1) (n + 2) ≫ g'' = g' ≫ I.cocomplex.d (n + 1) (n + 2) :=
  ⟨(J.exact_succ n).descToInjective
                                             /-
                                               C : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                               inst✝ : CategoryTheory.Abelian C
                                               Y Z : C
                                               I : CategoryTheory.InjectiveResolution Y
                                               J : CategoryTheory.InjectiveResolution Z
                                               n : Nat
                                               g : Quiver.Hom (J.cocomplex.X n) (I.cocomplex.X n)
                                               g' : Quiver.Hom (J.cocomplex.X (HAdd.hAdd n 1)) (I.cocomplex.X (HAdd.hAdd n 1))
                                               w : Eq (CategoryTheory.CategoryStruct.comp (J.cocomplex.d n (HAdd.hAdd n 1)) g …
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.mk (J.co …
                                             -/
    (g' ≫ I.cocomplex.d (n + 1) (n + 2)) (by simp [reassoc_of% w]),
                                             /-
                                               🎉 no goals
                                             -/
      (J.exact_succ n).comp_descToInjective _ _⟩


/-- A morphism in `C` descends to a chain map between injective resolutions. -/
def desc {Y Z : C} (f : Z ⟶ Y) (I : InjectiveResolution Y) (J : InjectiveResolution Z) :
    J.cocomplex ⟶ I.cocomplex :=
  CochainComplex.mkHom _ _ (descFZero f _ _) (descFOne f _ _) (descFOne_zero_comm f I J).symm
    fun n ⟨g, g', w⟩ => ⟨(descFSucc I J n g g' w.symm).1, (descFSucc I J n g g' w.symm).2.symm⟩


/-- The resolution maps intertwine the descent of a morphism and that morphism. -/
@[reassoc (attr := simp)]
theorem desc_commutes {Y Z : C} (f : Z ⟶ Y) (I : InjectiveResolution Y)
    (J : InjectiveResolution Z) : J.ι ≫ desc f I J = (CochainComplex.single₀ C).map f ≫ I.ι := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    Y Z : C
    f : Quiver.Hom Z Y
    I : CategoryTheory.InjectiveResolution Y
    J : CategoryTheory.InjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp J.ι (CategoryTheory.InjectiveResoluti …
  -/
  ext
  /-
    case hfg
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    Y Z : C
    f : Quiver.Hom Z Y
    I : CategoryTheory.InjectiveResolution Y
    J : CategoryTheory.InjectiveResolution Z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp J.ι (CategoryTheory.InjectiveResolut …
  -/
  simp [desc, descFOne, descFZero]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma desc_commutes_zero {Y Z : C} (f : Z ⟶ Y)
    (I : InjectiveResolution Y) (J : InjectiveResolution Z) :
    J.ι.f 0 ≫ (desc f I J).f 0 = f ≫ I.ι.f 0 :=
                                                                   /-
                                                                     C : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                     inst✝ : CategoryTheory.Abelian C
                                                                     Y Z : C
                                                                     f : Quiver.Hom Z Y
                                                                     I : CategoryTheory.InjectiveResolution Y
                                                                     J : CategoryTheory.InjectiveResolution Z
                                                                     ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CochainComplex.single₀ C).map f) I …
                                                                   -/
  (HomologicalComplex.congr_hom (desc_commutes f I J) 0).trans (by simp)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

-- Now that we've checked this property of the descent, we can seal away the actual definition.

/-- An auxiliary definition for `descHomotopyZero`. -/
def descHomotopyZeroZero {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (comm : I.ι ≫ f = 0) : I.cocomplex.X 1 ⟶ J.cocomplex.X 0 :=
  I.exact₀.descToInjective (f.f 0) (congr_fun (congr_arg HomologicalComplex.Hom.f comm) 0)


@[reassoc (attr := simp)]
lemma comp_descHomotopyZeroZero {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (comm : I.ι ≫ f = 0) :
    I.cocomplex.d 0 1 ≫ descHomotopyZeroZero f comm = f.f 0 :=
  I.exact₀.comp_descToInjective  _ _


/-- An auxiliary definition for `descHomotopyZero`. -/
def descHomotopyZeroOne {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (comm : I.ι ≫ f = (0 : _ ⟶ J.cocomplex)) :
    I.cocomplex.X 2 ⟶ J.cocomplex.X 1 :=
  (I.exact_succ 0).descToInjective (f.f 1 - descHomotopyZeroZero f comm ≫ J.cocomplex.d 0 1)
    (by rw [Preadditive.comp_sub, comp_descHomotopyZeroZero_assoc f comm,
          HomologicalComplex.Hom.comm, sub_self])


@[reassoc (attr := simp)]
lemma comp_descHomotopyZeroOne {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (comm : I.ι ≫ f = (0 : _ ⟶ J.cocomplex)) :
    I.cocomplex.d 1 2 ≫ descHomotopyZeroOne f comm =
      f.f 1 - descHomotopyZeroZero f comm ≫ J.cocomplex.d 0 1 :=
  (I.exact_succ 0).comp_descToInjective _ _


/-- An auxiliary definition for `descHomotopyZero`. -/
def descHomotopyZeroSucc {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (n : ℕ) (g : I.cocomplex.X (n + 1) ⟶ J.cocomplex.X n)
    (g' : I.cocomplex.X (n + 2) ⟶ J.cocomplex.X (n + 1))
    (w : f.f (n + 1) = I.cocomplex.d (n + 1) (n + 2) ≫ g' + g ≫ J.cocomplex.d n (n + 1)) :
    I.cocomplex.X (n + 3) ⟶ J.cocomplex.X (n + 2) :=
  (I.exact_succ (n + 1)).descToInjective (f.f (n + 2) - g' ≫ J.cocomplex.d _ _) (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        Y Z : C
        I : CategoryTheory.InjectiveResolution Y
        J : CategoryTheory.InjectiveResolution Z
        f : Quiver.Hom I.cocomplex J.cocomplex
        n : Nat
        g : Quiver.Hom (I.cocomplex.X (HAdd.hAdd n 1)) (J.cocomplex.X n)
        g' : Quiver.Hom (I.cocomplex.X (HAdd.hAdd n 2)) (J.cocomplex.X (HAdd.hAdd n 1))
        w : Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (I …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.mk (I.co …
      -/
      dsimp
      rw [Preadditive.comp_sub, ← HomologicalComplex.Hom.comm, w, Preadditive.add_comp,
        Category.assoc, Category.assoc, HomologicalComplex.d_comp_d, comp_zero,
        add_zero, sub_self])


@[reassoc (attr := simp)]
lemma comp_descHomotopyZeroSucc {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (n : ℕ) (g : I.cocomplex.X (n + 1) ⟶ J.cocomplex.X n)
    (g' : I.cocomplex.X (n + 2) ⟶ J.cocomplex.X (n + 1))
    (w : f.f (n + 1) = I.cocomplex.d (n + 1) (n + 2) ≫ g' + g ≫ J.cocomplex.d n (n + 1)) :
    I.cocomplex.d (n+2) (n+3) ≫ descHomotopyZeroSucc f n g g' w =
      f.f (n + 2) - g' ≫ J.cocomplex.d _ _ :=
  (I.exact_succ (n+1)).comp_descToInjective  _ _


/-- Any descent of the zero morphism is homotopic to zero. -/
def descHomotopyZero {Y Z : C} {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (f : I.cocomplex ⟶ J.cocomplex) (comm : I.ι ≫ f = 0) : Homotopy f 0 :=
                                                             /-
                                                               C : Type u
                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                               inst✝ : CategoryTheory.Abelian C
                                                               Y Z : C
                                                               I : CategoryTheory.InjectiveResolution Y
                                                               J : CategoryTheory.InjectiveResolution Z
                                                               f : Quiver.Hom I.cocomplex J.cocomplex
                                                               comm : Eq (CategoryTheory.CategoryStruct.comp I.ι f) 0
                                                               ⊢ Eq (f.f 0) (CategoryTheory.CategoryStruct.comp (I.cocomplex.d 0 1) (Category …
                                                             -/
  Homotopy.mkCoinductive _ (descHomotopyZeroZero f comm) (by simp)
                                                             /-
                                                               🎉 no goals
                                                             -/
                                     /-
                                       C : Type u
                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                       inst✝ : CategoryTheory.Abelian C
                                       Y Z : C
                                       I : CategoryTheory.InjectiveResolution Y
                                       J : CategoryTheory.InjectiveResolution Z
                                       f : Quiver.Hom I.cocomplex J.cocomplex
                                       comm : Eq (CategoryTheory.CategoryStruct.comp I.ι f) 0
                                       ⊢ Eq (f.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (CategoryTheory.In …
                                     -/
    (descHomotopyZeroOne f comm) (by simp) (fun n ⟨g, g', w⟩ =>
                                     /-
                                       🎉 no goals
                                     -/
                                       /-
                                         C : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                         inst✝ : CategoryTheory.Abelian C
                                         Y Z : C
                                         I : CategoryTheory.InjectiveResolution Y
                                         J : CategoryTheory.InjectiveResolution Z
                                         f : Quiver.Hom I.cocomplex J.cocomplex
                                         comm : Eq (CategoryTheory.CategoryStruct.comp I.ι f) 0
                                         n : Nat
                                         x✝ : PSigma fun f_1 => PSigma fun f' => Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (C …
                                         g : Quiver.Hom (I.cocomplex.X (HAdd.hAdd n 1)) (J.cocomplex.X n)
                                         g' : Quiver.Hom (I.cocomplex.X (HAdd.hAdd n 2)) (J.cocomplex.X (HAdd.hAdd n 1))
                                         w : Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp g  …
                                         ⊢ Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (I.c …
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
    ⟨descHomotopyZeroSucc f n g g' (by simp only [w, add_comm]), by simp⟩)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- Two descents of the same morphism are homotopic. -/
def descHomotopy {Y Z : C} (f : Y ⟶ Z) {I : InjectiveResolution Y} {J : InjectiveResolution Z}
    (g h : I.cocomplex ⟶ J.cocomplex) (g_comm : I.ι ≫ g = (CochainComplex.single₀ C).map f ≫ J.ι)
    (h_comm : I.ι ≫ h = (CochainComplex.single₀ C).map f ≫ J.ι) : Homotopy g h :=
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         inst✝ : CategoryTheory.Abelian C
                                                         Y Z : C
                                                         f : Quiver.Hom Y Z
                                                         I : CategoryTheory.InjectiveResolution Y
                                                         J : CategoryTheory.InjectiveResolution Z
                                                         g h : Quiver.Hom I.cocomplex J.cocomplex
                                                         g_comm : Eq (CategoryTheory.CategoryStruct.comp I.ι g) (CategoryTheory.Categor …
                                                         h_comm : Eq (CategoryTheory.CategoryStruct.comp I.ι h) (CategoryTheory.Categor …
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp I.ι (HSub.hSub g h)) 0
                                                       -/
  Homotopy.equivSubZero.invFun (descHomotopyZero _ (by simp [g_comm, h_comm]))
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The descent of the identity morphism is homotopic to the identity cochain map. -/
def descIdHomotopy (X : C) (I : InjectiveResolution X) :
    Homotopy (desc (𝟙 X) I I) (𝟙 I.cocomplex) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X : C
    I : CategoryTheory.InjectiveResolution X
    ⊢ Homotopy (CategoryTheory.InjectiveResolution.desc (CategoryTheory.CategorySt …
  -/
                               /-
                                 🎉 no goals
                               -/
  apply descHomotopy (𝟙 X) <;> simp
                               /-
                                 🎉 no goals
                               -/


/-- The descent of a composition is homotopic to the composition of the descents. -/
def descCompHomotopy {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (I : InjectiveResolution X)
    (J : InjectiveResolution Y) (K : InjectiveResolution Z) :
    Homotopy (desc (f ≫ g) K I) (desc f J I ≫ desc g K J) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    K : CategoryTheory.InjectiveResolution Z
    ⊢ Homotopy (CategoryTheory.InjectiveResolution.desc (CategoryTheory.CategorySt …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  apply descHomotopy (f ≫ g) <;> simp
                                 /-
                                   🎉 no goals
                                 -/

-- We don't care about the actual definitions of these homotopies.

/-- Any two injective resolutions are homotopy equivalent. -/
def homotopyEquiv {X : C} (I J : InjectiveResolution X) :
    HomotopyEquiv I.cocomplex J.cocomplex where
  hom := desc (𝟙 X) J I
  inv := desc (𝟙 X) I J
  homotopyHomInvId := (descCompHomotopy (𝟙 X) (𝟙 X) I J I).symm.trans <| by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      I J : CategoryTheory.InjectiveResolution X
      ⊢ Homotopy (CategoryTheory.InjectiveResolution.desc (CategoryTheory.CategorySt …
    -/
    simpa [id_comp] using descIdHomotopy _ _
    /-
      🎉 no goals
    -/
  homotopyInvHomId := (descCompHomotopy (𝟙 X) (𝟙 X) J I J).symm.trans <| by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      I J : CategoryTheory.InjectiveResolution X
      ⊢ Homotopy (CategoryTheory.InjectiveResolution.desc (CategoryTheory.CategorySt …
    -/
    simpa [id_comp] using descIdHomotopy _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem homotopyEquiv_hom_ι {X : C} (I J : InjectiveResolution X) :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Abelian C
                                                X : C
                                                I J : CategoryTheory.InjectiveResolution X
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp I.ι (I.homotopyEquiv J).hom) J.ι
                                              -/
    I.ι ≫ (homotopyEquiv I J).hom = J.ι := by simp [homotopyEquiv]
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc (attr := simp)]
theorem homotopyEquiv_inv_ι {X : C} (I J : InjectiveResolution X) :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Abelian C
                                                X : C
                                                I J : CategoryTheory.InjectiveResolution X
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp J.ι (I.homotopyEquiv J).inv) I.ι
                                              -/
    J.ι ≫ (homotopyEquiv I J).inv = I.ι := by simp [homotopyEquiv]
                                              /-
                                                🎉 no goals
                                              -/


/-- An arbitrarily chosen injective resolution of an object. -/
abbrev injectiveResolution (Z : C) [HasInjectiveResolution Z] : InjectiveResolution Z :=
  (HasInjectiveResolution.out (Z := Z)).some


/-- Taking injective resolutions is functorial,
if considered with target the homotopy category
(`ℕ`-indexed cochain complexes and chain maps up to homotopy).
-/
def injectiveResolutions : C ⥤ HomotopyCategory C (ComplexShape.up ℕ) where
  obj X := (HomotopyCategory.quotient _ _).obj (injectiveResolution X).cocomplex
  map f := (HomotopyCategory.quotient _ _).map (InjectiveResolution.desc f _ _)
  map_id X := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasInjectiveResolutions C
      X : C
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.up Nat)).ob …
    -/
    rw [← (HomotopyCategory.quotient _ _).map_id]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasInjectiveResolutions C
      X : C
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.up Nat)).ob …
    -/
    apply HomotopyCategory.eq_of_homotopy
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasInjectiveResolutions C
      X : C
      ⊢ Homotopy (CategoryTheory.InjectiveResolution.desc (CategoryTheory.CategorySt …
    -/
    apply InjectiveResolution.descIdHomotopy
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasInjectiveResolutions C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.up Nat)).ob …
    -/
    rw [← (HomotopyCategory.quotient _ _).map_comp]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasInjectiveResolutions C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.up Nat)).ob …
    -/
    apply HomotopyCategory.eq_of_homotopy
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasInjectiveResolutions C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Homotopy (CategoryTheory.InjectiveResolution.desc (CategoryTheory.CategorySt …
    -/
    apply InjectiveResolution.descCompHomotopy
    /-
      🎉 no goals
    -/

/-- If `I : InjectiveResolution X`, then the chosen `(injectiveResolutions C).obj X`
is isomorphic (in the homotopy category) to `I.cocomplex`. -/
def InjectiveResolution.iso {X : C} (I : InjectiveResolution X) :
    (injectiveResolutions C).obj X ≅
      (HomotopyCategory.quotient _ _).obj I.cocomplex :=
  HomotopyCategory.isoOfHomotopyEquiv (homotopyEquiv _ _)


@[reassoc]
lemma InjectiveResolution.iso_hom_naturality {X Y : C} (f : X ⟶ Y)
    (I : InjectiveResolution X) (J : InjectiveResolution Y)
    (φ : I.cocomplex ⟶ J.cocomplex) (comm : I.ι.f 0 ≫ φ.f 0 = f ≫ J.ι.f 0) :
    (injectiveResolutions C).map f ≫ J.iso.hom =
      I.iso.hom ≫ (HomotopyCategory.quotient _ _).map φ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasInjectiveResolutions C
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.injectiveResolutions …
  -/
  apply HomotopyCategory.eq_of_homotopy
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasInjectiveResolutions C
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    ⊢ Homotopy (CategoryTheory.CategoryStruct.comp (CategoryTheory.InjectiveResolu …
  -/
  apply descHomotopy f
  /-
    case h.g_comm
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasInjectiveResolutions C
    X Y : C
    f : Quiver.Hom X Y
    I : CategoryTheory.InjectiveResolution X
    J : CategoryTheory.InjectiveResolution Y
    φ : Quiver.Hom I.cocomplex J.cocomplex
    comm : Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (φ.f 0)) (CategoryTheo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.injectiveResolution X …
  -/
  all_goals aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc]
lemma InjectiveResolution.iso_inv_naturality {X Y : C} (f : X ⟶ Y)
    (I : InjectiveResolution X) (J : InjectiveResolution Y)
    (φ : I.cocomplex ⟶ J.cocomplex) (comm : I.ι.f 0 ≫ φ.f 0 = f ≫ J.ι.f 0) :
    I.iso.inv ≫ (injectiveResolutions C).map f =
      (HomotopyCategory.quotient _ _).map φ ≫ J.iso.inv := by
  rw [← cancel_mono (J.iso).hom, Category.assoc, iso_hom_naturality f I J φ comm,
    Iso.inv_hom_id_assoc, Category.assoc, Iso.inv_hom_id, Category.comp_id]


theorem exact_f_d {X Y : C} (f : X ⟶ Y) :
                                 /-
                                   C : Type u
                                   inst✝² : CategoryTheory.Category.{v, u} C
                                   inst✝¹ : CategoryTheory.Abelian C
                                   inst✝ : CategoryTheory.EnoughInjectives C
                                   X Y : C
                                   f : Quiver.Hom X Y
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Injective.d f)) 0
                                 -/
    (ShortComplex.mk f (d f) (by simp)).Exact := by
                                 /-
                                   🎉 no goals
                                 -/
  let α : ShortComplex.mk f (cokernel.π f) (by simp) ⟶ ShortComplex.mk f (d f) (by simp) :=
    { τ₁ := 𝟙 _
      τ₂ := 𝟙 _
      τ₃ := Injective.ι _  }
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokern …
    ⊢ (CategoryTheory.ShortComplex.mk f (CategoryTheory.Injective.d f) ⋯).Exact
  -/
  have : Epi α.τ₁ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokern …
    this : CategoryTheory.Epi α.τ₁
    ⊢ (CategoryTheory.ShortComplex.mk f (CategoryTheory.Injective.d f) ⋯).Exact
  -/
  have : IsIso α.τ₂ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokern …
    this✝ : CategoryTheory.Epi α.τ₁
    this : CategoryTheory.IsIso α.τ₂
    ⊢ (CategoryTheory.ShortComplex.mk f (CategoryTheory.Injective.d f) ⋯).Exact
  -/
  have : Mono α.τ₃ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokern …
    this✝¹ : CategoryTheory.Epi α.τ₁
    this✝ : CategoryTheory.IsIso α.τ₂
    this : CategoryTheory.Mono α.τ₃
    ⊢ (CategoryTheory.ShortComplex.mk f (CategoryTheory.Injective.d f) ⋯).Exact
  -/
  rw [← ShortComplex.exact_iff_of_epi_of_isIso_of_mono α]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokern …
    this✝¹ : CategoryTheory.Epi α.τ₁
    this✝ : CategoryTheory.IsIso α.τ₂
    this : CategoryTheory.Mono α.τ₃
    ⊢ (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokernel.π f) ⋯).Ex …
  -/
  apply ShortComplex.exact_of_g_is_cokernel
  /-
    case hS
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk f (CategoryTheory.Limits.cokern …
    this✝¹ : CategoryTheory.Epi α.τ₁
    this✝ : CategoryTheory.IsIso α.τ₂
    this : CategoryTheory.Mono α.τ₃
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ (C …
  -/
  apply cokernelIsCokernel
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `InjectiveResolution.of`. -/
def ofCocomplex : CochainComplex C ℕ :=
  CochainComplex.mk' (Injective.under Z) (Injective.syzygies (Injective.ι Z))
                                                                 /-
                                                                   C : Type u
                                                                   inst✝² : CategoryTheory.Category.{v, u} C
                                                                   inst✝¹ : CategoryTheory.Abelian C
                                                                   inst✝ : CategoryTheory.EnoughInjectives C
                                                                   Z X₀✝ X₁✝ : C
                                                                   f : Quiver.Hom X₀✝ X₁✝
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Injective.d f)) 0
                                                                 -/
    (Injective.d (Injective.ι Z)) fun f => ⟨_, Injective.d f, by simp⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma ofCocomplex_d_0_1 :
    (ofCocomplex Z).d 0 1 = d (Injective.ι Z) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    Z : C
    ⊢ Eq ((CategoryTheory.InjectiveResolution.ofCocomplex Z).d 0 1) (CategoryTheor …
  -/
  simp [ofCocomplex]
  /-
    🎉 no goals
  -/


lemma ofCocomplex_exactAt_succ (n : ℕ) :
    (ofCocomplex Z).ExactAt (n + 1) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    Z : C
    n : Nat
    ⊢ HomologicalComplex.ExactAt (CategoryTheory.InjectiveResolution.ofCocomplex Z …
  -/
  rw [HomologicalComplex.exactAt_iff' _ n (n + 1) (n + 1 + 1) (by simp) (by simp)]
  dsimp [ofCocomplex, CochainComplex.mk', CochainComplex.mk, HomologicalComplex.sc',
      HomologicalComplex.shortComplexFunctor']
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    Z : C
    n : Nat
    ⊢ (CategoryTheory.ShortComplex.mk ((CochainComplex.of (fun n => (CochainComple …
  -/
  simp only [CochainComplex.of_d]
  match n with
  | 0 => apply exact_f_d ((CochainComplex.mkAux _ _ _
      (d (Injective.ι Z)) (d (d (Injective.ι Z))) _ _ 0).f)
  | n+1 => apply exact_f_d ((CochainComplex.mkAux _ _ _
      (d (Injective.ι Z)) (d (d (Injective.ι Z))) _ _ (n+1)).f)


instance (n : ℕ) : Injective ((ofCocomplex Z).X n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughInjectives C
    Z : C
    n : Nat
    ⊢ CategoryTheory.Injective ((CategoryTheory.InjectiveResolution.ofCocomplex Z) …
  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  obtain (_ | _ | _ | n) := n <;> apply Injective.injective_under
                                  /-
                                    🎉 no goals
                                  -/


/-- In any abelian category with enough injectives,
`InjectiveResolution.of Z` constructs an injective resolution of the object `Z`.
-/
irreducible_def of : InjectiveResolution Z where
  cocomplex := ofCocomplex Z
  ι := (CochainComplex.fromSingle₀Equiv _ _).symm ⟨Injective.ι Z,
       /-
         C : Type u
         inst✝² : CategoryTheory.Category.{v, u} C
         inst✝¹ : CategoryTheory.Abelian C
         inst✝ : CategoryTheory.EnoughInjectives C
         Z : C
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Injective.ι Z) ((Cate …
       -/
    by rw [ofCocomplex_d_0_1, cokernel.condition_assoc, zero_comp]⟩
       /-
         🎉 no goals
       -/
  quasiIso := ⟨fun n => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.EnoughInjectives C
      Z : C
      n : Nat
      ⊢ QuasiIsoAt (((CategoryTheory.InjectiveResolution.ofCocomplex Z).fromSingle₀E …
    -/
    cases n
      /-
        case zero
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Abelian C
        inst✝ : CategoryTheory.EnoughInjectives C
        Z : C
        ⊢ QuasiIsoAt (((CategoryTheory.InjectiveResolution.ofCocomplex Z).fromSingle₀E …
      -/
    · rw [CochainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros]
      · refine (ShortComplex.exact_and_mono_f_iff_of_iso ?_).2
          ⟨exact_f_d (Injective.ι Z), by dsimp; infer_instance⟩
        exact ShortComplex.isoMk (Iso.refl _) (Iso.refl _) (Iso.refl _) (by simp)
          (by simp [ofCocomplex])
      /-
        case zero.hf₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Abelian C
        inst✝ : CategoryTheory.EnoughInjectives C
        Z : C
        ⊢ Eq ((HomologicalComplex.shortComplexFunctor' C (ComplexShape.up Nat) 0 0 1). …
      -/
      all_goals rfl
      /-
        🎉 no goals
      -/
      /-
        case succ
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Abelian C
        inst✝ : CategoryTheory.EnoughInjectives C
        Z : C
        n✝ : Nat
        ⊢ QuasiIsoAt (((CategoryTheory.InjectiveResolution.ofCocomplex Z).fromSingle₀E …
      -/
    · rw [quasiIsoAt_iff_exactAt]
        /-
          case succ
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.EnoughInjectives C
          Z : C
          n✝ : Nat
          ⊢ HomologicalComplex.ExactAt (CategoryTheory.InjectiveResolution.ofCocomplex Z …
        -/
      · apply ofCocomplex_exactAt_succ
        /-
          🎉 no goals
        -/
        /-
          case succ.hK
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.EnoughInjectives C
          Z : C
          n✝ : Nat
          ⊢ HomologicalComplex.ExactAt ((CochainComplex.single₀ C).obj Z) (HAdd.hAdd n✝ 1)
        -/
      · apply CochainComplex.exactAt_succ_single_obj⟩
        /-
          🎉 no goals
        -/


instance (priority := 100) (Z : C) : HasInjectiveResolution Z where out := ⟨of Z⟩


instance (priority := 100) : HasInjectiveResolutions C where out _ := inferInstance


