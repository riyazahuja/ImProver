/-- Auxiliary construction for `lift`. -/
def liftFZero {Y Z : C} (f : Y ⟶ Z) (P : ProjectiveResolution Y) (Q : ProjectiveResolution Z) :
    P.complex.X 0 ⟶ Q.complex.X 0 :=
  Projective.factorThru (P.π.f 0 ≫ f) (Q.π.f 0)


lemma exact₀ {Z : C} (P : ProjectiveResolution Z) :
    (ShortComplex.mk _ _ P.complex_d_comp_π_f_zero).Exact :=
  ShortComplex.exact_of_g_is_cokernel _ P.isColimitCokernelCofork


/-- Auxiliary construction for `lift`. -/
def liftFOne {Y Z : C} (f : Y ⟶ Z) (P : ProjectiveResolution Y) (Q : ProjectiveResolution Z) :
    P.complex.X 1 ⟶ Q.complex.X 1 :=
                                                                      /-
                                                                        C : Type u
                                                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                        inst✝ : CategoryTheory.Abelian C
                                                                        Y Z : C
                                                                        f : Quiver.Hom Y Z
                                                                        P : CategoryTheory.ProjectiveResolution Y
                                                                        Q : CategoryTheory.ProjectiveResolution Z
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                                      -/
  Q.exact₀.liftFromProjective (P.complex.d 1 0 ≫ liftFZero f P Q) (by simp [liftFZero])
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem liftFOne_zero_comm {Y Z : C} (f : Y ⟶ Z) (P : ProjectiveResolution Y)
    (Q : ProjectiveResolution Z) :
    liftFOne f P Q ≫ Q.complex.d 1 0 = P.complex.d 1 0 ≫ liftFZero f P Q := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    Y Z : C
    f : Quiver.Hom Y Z
    P : CategoryTheory.ProjectiveResolution Y
    Q : CategoryTheory.ProjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ProjectiveResolution. …
  -/
  apply Q.exact₀.liftFromProjective_comp
  /-
    🎉 no goals
  -/


/-- Auxiliary construction for `lift`. -/
def liftFSucc {Y Z : C} (P : ProjectiveResolution Y) (Q : ProjectiveResolution Z) (n : ℕ)
    (g : P.complex.X n ⟶ Q.complex.X n) (g' : P.complex.X (n + 1) ⟶ Q.complex.X (n + 1))
    (w : g' ≫ Q.complex.d (n + 1) n = P.complex.d (n + 1) n ≫ g) :
    Σ'g'' : P.complex.X (n + 2) ⟶ Q.complex.X (n + 2),
      g'' ≫ Q.complex.d (n + 2) (n + 1) = P.complex.d (n + 2) (n + 1) ≫ g' :=
  ⟨(Q.exact_succ n).liftFromProjective
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             inst✝ : CategoryTheory.Abelian C
                                             Y Z : C
                                             P : CategoryTheory.ProjectiveResolution Y
                                             Q : CategoryTheory.ProjectiveResolution Z
                                             n : Nat
                                             g : Quiver.Hom (P.complex.X n) (Q.complex.X n)
                                             g' : Quiver.Hom (P.complex.X (HAdd.hAdd n 1)) (Q.complex.X (HAdd.hAdd n 1))
                                             w : Eq (CategoryTheory.CategoryStruct.comp g' (Q.complex.d (HAdd.hAdd n 1) n)) …
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                           -/
    (P.complex.d (n + 2) (n + 1) ≫ g') (by simp [w]),
                                           /-
                                             🎉 no goals
                                           -/
      (Q.exact_succ n).liftFromProjective_comp _ _⟩


/-- A morphism in `C` lift to a chain map between projective resolutions. -/
def lift {Y Z : C} (f : Y ⟶ Z) (P : ProjectiveResolution Y) (Q : ProjectiveResolution Z) :
    P.complex ⟶ Q.complex :=
  ChainComplex.mkHom _ _ (liftFZero f _ _) (liftFOne f _ _) (liftFOne_zero_comm f P Q)
    fun n ⟨g, g', w⟩ => ⟨(liftFSucc P Q n g g' w).1, (liftFSucc P Q n g g' w).2⟩


/-- The resolution maps intertwine the lift of a morphism and that morphism. -/
@[reassoc (attr := simp)]
theorem lift_commutes {Y Z : C} (f : Y ⟶ Z) (P : ProjectiveResolution Y)
    (Q : ProjectiveResolution Z) : lift f P Q ≫ Q.π = P.π ≫ (ChainComplex.single₀ C).map f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    Y Z : C
    f : Quiver.Hom Y Z
    P : CategoryTheory.ProjectiveResolution Y
    Q : CategoryTheory.ProjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ProjectiveResolution. …
  -/
  ext
  /-
    case hfg
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    Y Z : C
    f : Quiver.Hom Y Z
    P : CategoryTheory.ProjectiveResolution Y
    Q : CategoryTheory.ProjectiveResolution Z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.ProjectiveResolution …
  -/
  simp [lift, liftFZero, liftFOne]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma lift_commutes_zero {Y Z : C} (f : Y ⟶ Z)
    (P : ProjectiveResolution Y) (Q : ProjectiveResolution Z) :
    (lift f P Q).f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f :=
                                                                   /-
                                                                     C : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                     inst✝ : CategoryTheory.Abelian C
                                                                     Y Z : C
                                                                     f : Quiver.Hom Y Z
                                                                     P : CategoryTheory.ProjectiveResolution Y
                                                                     Q : CategoryTheory.ProjectiveResolution Z
                                                                     ⊢ Eq ((CategoryTheory.CategoryStruct.comp P.π ((ChainComplex.single₀ C).map f) …
                                                                   -/
  (HomologicalComplex.congr_hom (lift_commutes f P Q) 0).trans (by simp)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- An auxiliary definition for `liftHomotopyZero`. -/
def liftHomotopyZeroZero {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (comm : f ≫ Q.π = 0) : P.complex.X 0 ⟶ Q.complex.X 1 :=
  Q.exact₀.liftFromProjective (f.f 0) (congr_fun (congr_arg HomologicalComplex.Hom.f comm) 0)


@[reassoc (attr := simp)]
lemma liftHomotopyZeroZero_comp {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (comm : f ≫ Q.π = 0) :
    liftHomotopyZeroZero f comm ≫ Q.complex.d 1 0 = f.f 0 :=
  Q.exact₀.liftFromProjective_comp  _ _


/-- An auxiliary definition for `liftHomotopyZero`. -/
def liftHomotopyZeroOne {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (comm : f ≫ Q.π = 0) :
    P.complex.X 1 ⟶ Q.complex.X 2 :=
  (Q.exact_succ 0).liftFromProjective (f.f 1 - P.complex.d 1 0 ≫ liftHomotopyZeroZero f comm)
    (by rw [Preadditive.sub_comp, assoc, HomologicalComplex.Hom.comm,
              liftHomotopyZeroZero_comp, sub_self])


@[reassoc (attr := simp)]
lemma liftHomotopyZeroOne_comp {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (comm : f ≫ Q.π = 0) :
    liftHomotopyZeroOne f comm ≫ Q.complex.d 2 1 =
      f.f 1 - P.complex.d 1 0 ≫ liftHomotopyZeroZero f comm :=
  (Q.exact_succ 0).liftFromProjective_comp _ _


/-- An auxiliary definition for `liftHomotopyZero`. -/
def liftHomotopyZeroSucc {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (n : ℕ) (g : P.complex.X n ⟶ Q.complex.X (n + 1))
    (g' : P.complex.X (n + 1) ⟶ Q.complex.X (n + 2))
    (w : f.f (n + 1) = P.complex.d (n + 1) n ≫ g + g' ≫ Q.complex.d (n + 2) (n + 1)) :
    P.complex.X (n + 2) ⟶ Q.complex.X (n + 3) :=
                                                                                     /-
                                                                                       C : Type u
                                                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                       inst✝ : CategoryTheory.Abelian C
                                                                                       Y Z : C
                                                                                       P : CategoryTheory.ProjectiveResolution Y
                                                                                       Q : CategoryTheory.ProjectiveResolution Z
                                                                                       f : Quiver.Hom P.complex Q.complex
                                                                                       n : Nat
                                                                                       g : Quiver.Hom (P.complex.X n) (Q.complex.X (HAdd.hAdd n 1))
                                                                                       g' : Quiver.Hom (P.complex.X (HAdd.hAdd n 1)) (Q.complex.X (HAdd.hAdd n 2))
                                                                                       w : Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P …
                                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (f.f (HAdd.hAdd n 2)) (Cat …
                                                                                     -/
  (Q.exact_succ (n + 1)).liftFromProjective (f.f (n + 2) - P.complex.d _ _ ≫ g') (by simp [w])
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[reassoc (attr := simp)]
lemma liftHomotopyZeroSucc_comp {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (n : ℕ) (g : P.complex.X n ⟶ Q.complex.X (n + 1))
    (g' : P.complex.X (n + 1) ⟶ Q.complex.X (n + 2))
    (w : f.f (n + 1) = P.complex.d (n + 1) n ≫ g + g' ≫ Q.complex.d (n + 2) (n + 1)) :
    liftHomotopyZeroSucc f n g g' w ≫ Q.complex.d (n + 3) (n + 2) =
      f.f (n + 2) - P.complex.d _ _ ≫ g' :=
  (Q.exact_succ (n+1)).liftFromProjective_comp  _ _


/-- Any lift of the zero morphism is homotopic to zero. -/
def liftHomotopyZero {Y Z : C} {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (f : P.complex ⟶ Q.complex) (comm : f ≫ Q.π = 0) : Homotopy f 0 :=
                                                           /-
                                                             C : Type u
                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                             inst✝ : CategoryTheory.Abelian C
                                                             Y Z : C
                                                             P : CategoryTheory.ProjectiveResolution Y
                                                             Q : CategoryTheory.ProjectiveResolution Z
                                                             f : Quiver.Hom P.complex Q.complex
                                                             comm : Eq (CategoryTheory.CategoryStruct.comp f Q.π) 0
                                                             ⊢ Eq (f.f 0) (CategoryTheory.CategoryStruct.comp (CategoryTheory.ProjectiveRes …
                                                           -/
  Homotopy.mkInductive _ (liftHomotopyZeroZero f comm) (by simp )
                                                           /-
                                                             🎉 no goals
                                                           -/
                                     /-
                                       C : Type u
                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                       inst✝ : CategoryTheory.Abelian C
                                       Y Z : C
                                       P : CategoryTheory.ProjectiveResolution Y
                                       Q : CategoryTheory.ProjectiveResolution Z
                                       f : Quiver.Hom P.complex Q.complex
                                       comm : Eq (CategoryTheory.CategoryStruct.comp f Q.π) 0
                                       ⊢ Eq (f.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.complex.d 1 0)  …
                                     -/
    (liftHomotopyZeroOne f comm) (by simp) fun n ⟨g, g', w⟩ =>
                                     /-
                                       🎉 no goals
                                     -/
                                         /-
                                           C : Type u
                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                           inst✝ : CategoryTheory.Abelian C
                                           Y Z : C
                                           P : CategoryTheory.ProjectiveResolution Y
                                           Q : CategoryTheory.ProjectiveResolution Z
                                           f : Quiver.Hom P.complex Q.complex
                                           comm : Eq (CategoryTheory.CategoryStruct.comp f Q.π) 0
                                           n : Nat
                                           x✝ : PSigma fun f_1 => PSigma fun f' => Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (C …
                                           g : Quiver.Hom (P.complex.X n) (Q.complex.X (HAdd.hAdd n 1))
                                           g' : Quiver.Hom (P.complex.X (HAdd.hAdd n 1)) (Q.complex.X (HAdd.hAdd n 2))
                                           w : Eq (f.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P …
                                           ⊢ Eq (f.f (HAdd.hAdd n 2)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.c …
                                         -/
    ⟨liftHomotopyZeroSucc f n g g' w, by simp⟩
                                         /-
                                           🎉 no goals
                                         -/


/-- Two lifts of the same morphism are homotopic. -/
def liftHomotopy {Y Z : C} (f : Y ⟶ Z) {P : ProjectiveResolution Y} {Q : ProjectiveResolution Z}
    (g h : P.complex ⟶ Q.complex) (g_comm : g ≫ Q.π = P.π ≫ (ChainComplex.single₀ C).map f)
    (h_comm : h ≫ Q.π = P.π ≫ (ChainComplex.single₀ C).map f) : Homotopy g h :=
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         inst✝ : CategoryTheory.Abelian C
                                                         Y Z : C
                                                         f : Quiver.Hom Y Z
                                                         P : CategoryTheory.ProjectiveResolution Y
                                                         Q : CategoryTheory.ProjectiveResolution Z
                                                         g h : Quiver.Hom P.complex Q.complex
                                                         g_comm : Eq (CategoryTheory.CategoryStruct.comp g Q.π) (CategoryTheory.Categor …
                                                         h_comm : Eq (CategoryTheory.CategoryStruct.comp h Q.π) (CategoryTheory.Categor …
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub g h) Q.π) 0
                                                       -/
  Homotopy.equivSubZero.invFun (liftHomotopyZero _ (by simp [g_comm, h_comm]))
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The lift of the identity morphism is homotopic to the identity chain map. -/
def liftIdHomotopy (X : C) (P : ProjectiveResolution X) :
    Homotopy (lift (𝟙 X) P P) (𝟙 P.complex) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X : C
    P : CategoryTheory.ProjectiveResolution X
    ⊢ Homotopy (CategoryTheory.ProjectiveResolution.lift (CategoryTheory.CategoryS …
  -/
                               /-
                                 🎉 no goals
                               -/
  apply liftHomotopy (𝟙 X) <;> simp
                               /-
                                 🎉 no goals
                               -/


/-- The lift of a composition is homotopic to the composition of the lifts. -/
def liftCompHomotopy {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (P : ProjectiveResolution X)
    (Q : ProjectiveResolution Y) (R : ProjectiveResolution Z) :
    Homotopy (lift (f ≫ g) P R) (lift f P Q ≫ lift g Q R) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    R : CategoryTheory.ProjectiveResolution Z
    ⊢ Homotopy (CategoryTheory.ProjectiveResolution.lift (CategoryTheory.CategoryS …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  apply liftHomotopy (f ≫ g) <;> simp
                                 /-
                                   🎉 no goals
                                 -/

-- We don't care about the actual definitions of these homotopies.

/-- Any two projective resolutions are homotopy equivalent. -/
def homotopyEquiv {X : C} (P Q : ProjectiveResolution X) :
    HomotopyEquiv P.complex Q.complex where
  hom := lift (𝟙 X) P Q
  inv := lift (𝟙 X) Q P
  homotopyHomInvId := (liftCompHomotopy (𝟙 X) (𝟙 X) P Q P).symm.trans <| by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      P Q : CategoryTheory.ProjectiveResolution X
      ⊢ Homotopy (CategoryTheory.ProjectiveResolution.lift (CategoryTheory.CategoryS …
    -/
    simpa [id_comp] using liftIdHomotopy _ _
    /-
      🎉 no goals
    -/
  homotopyInvHomId := (liftCompHomotopy (𝟙 X) (𝟙 X) Q P Q).symm.trans <| by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      P Q : CategoryTheory.ProjectiveResolution X
      ⊢ Homotopy (CategoryTheory.ProjectiveResolution.lift (CategoryTheory.CategoryS …
    -/
    simpa [id_comp] using liftIdHomotopy _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem homotopyEquiv_hom_π {X : C} (P Q : ProjectiveResolution X) :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Abelian C
                                                X : C
                                                P Q : CategoryTheory.ProjectiveResolution X
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.homotopyEquiv Q).hom Q.π) P.π
                                              -/
    (homotopyEquiv P Q).hom ≫ Q.π = P.π := by simp [homotopyEquiv]
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc (attr := simp)]
theorem homotopyEquiv_inv_π {X : C} (P Q : ProjectiveResolution X) :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Abelian C
                                                X : C
                                                P Q : CategoryTheory.ProjectiveResolution X
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.homotopyEquiv Q).inv P.π) Q.π
                                              -/
    (homotopyEquiv P Q).inv ≫ P.π = Q.π := by simp [homotopyEquiv]
                                              /-
                                                🎉 no goals
                                              -/


/-- An arbitrarily chosen projective resolution of an object. -/
abbrev projectiveResolution (Z : C) [HasZeroObject C]
    [HasZeroMorphisms C] [HasProjectiveResolution Z] :
    ProjectiveResolution Z :=
  (HasProjectiveResolution.out (Z := Z)).some


/-- Taking projective resolutions is functorial,
if considered with target the homotopy category
(`ℕ`-indexed chain complexes and chain maps up to homotopy).
-/
def projectiveResolutions : C ⥤ HomotopyCategory C (ComplexShape.down ℕ) where
  obj X := (HomotopyCategory.quotient _ _).obj (projectiveResolution X).complex
  map f := (HomotopyCategory.quotient _ _).map (ProjectiveResolution.lift f _ _)
  map_id X := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasProjectiveResolutions C
      X : C
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.down Nat)). …
    -/
    rw [← (HomotopyCategory.quotient _ _).map_id]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasProjectiveResolutions C
      X : C
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.down Nat)). …
    -/
    apply HomotopyCategory.eq_of_homotopy
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasProjectiveResolutions C
      X : C
      ⊢ Homotopy (CategoryTheory.ProjectiveResolution.lift (CategoryTheory.CategoryS …
    -/
    apply ProjectiveResolution.liftIdHomotopy
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasProjectiveResolutions C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.down Nat)). …
    -/
    rw [← (HomotopyCategory.quotient _ _).map_comp]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasProjectiveResolutions C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => (HomotopyCategory.quotient C (ComplexShape.down Nat)). …
    -/
    apply HomotopyCategory.eq_of_homotopy
    /-
      case h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasProjectiveResolutions C
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Homotopy (CategoryTheory.ProjectiveResolution.lift (CategoryTheory.CategoryS …
    -/
    apply ProjectiveResolution.liftCompHomotopy
    /-
      🎉 no goals
    -/


/-- If `P : ProjectiveResolution X`, then the chosen `(projectiveResolutions C).obj X`
is isomorphic (in the homotopy category) to `P.complex`. -/
def ProjectiveResolution.iso {X : C} (P : ProjectiveResolution X) :
    (projectiveResolutions C).obj X ≅
      (HomotopyCategory.quotient _ _).obj P.complex :=
  HomotopyCategory.isoOfHomotopyEquiv (homotopyEquiv _ _)


@[reassoc]
lemma ProjectiveResolution.iso_inv_naturality {X Y : C} (f : X ⟶ Y)
    (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f) :
    P.iso.inv ≫ (projectiveResolutions C).map f =
      (HomotopyCategory.quotient _ _).map φ ≫ Q.iso.inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasProjectiveResolutions C
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp P.iso.inv ((CategoryTheory.projective …
  -/
  apply HomotopyCategory.eq_of_homotopy
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasProjectiveResolutions C
    X Y : C
    f : Quiver.Hom X Y
    P : CategoryTheory.ProjectiveResolution X
    Q : CategoryTheory.ProjectiveResolution Y
    φ : Quiver.Hom P.complex Q.complex
    comm : Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (Q.π.f 0)) (CategoryTheo …
    ⊢ Homotopy (CategoryTheory.CategoryStruct.comp ((CategoryTheory.projectiveReso …
  -/
  apply liftHomotopy f
  all_goals
    aesop_cat


@[reassoc]
lemma ProjectiveResolution.iso_hom_naturality {X Y : C} (f : X ⟶ Y)
    (P : ProjectiveResolution X) (Q : ProjectiveResolution Y)
    (φ : P.complex ⟶ Q.complex) (comm : φ.f 0 ≫ Q.π.f 0 = P.π.f 0 ≫ f) :
    (projectiveResolutions C).map f ≫ Q.iso.hom =
      P.iso.hom ≫ (HomotopyCategory.quotient _ _).map φ := by
  rw [← cancel_epi (P.iso).inv, iso_inv_naturality_assoc f P Q φ comm,
    Iso.inv_hom_id_assoc, Iso.inv_hom_id, comp_id]


variable {C} in
theorem exact_d_f {X Y : C} (f : X ⟶ Y) :
                                 /-
                                   C : Type u
                                   inst✝² : CategoryTheory.Category.{v, u} C
                                   inst✝¹ : CategoryTheory.Abelian C
                                   inst✝ : CategoryTheory.EnoughProjectives C
                                   X Y : C
                                   f : Quiver.Hom X Y
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Projective.d f) f) 0
                                 -/
    (ShortComplex.mk (d f) f (by simp)).Exact := by
                                 /-
                                   🎉 no goals
                                 -/
  let α : ShortComplex.mk (d f) f (by simp) ⟶ ShortComplex.mk (kernel.ι f) f (by simp) :=
    { τ₁ := Projective.π _
      τ₂ := 𝟙 _
      τ₃ := 𝟙 _ }
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) …
    ⊢ (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) f ⋯).Exact
  -/
  have : Epi α.τ₁ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) …
    this : CategoryTheory.Epi α.τ₁
    ⊢ (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) f ⋯).Exact
  -/
  have : IsIso α.τ₂ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) …
    this✝ : CategoryTheory.Epi α.τ₁
    this : CategoryTheory.IsIso α.τ₂
    ⊢ (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) f ⋯).Exact
  -/
  have : Mono α.τ₃ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) …
    this✝¹ : CategoryTheory.Epi α.τ₁
    this✝ : CategoryTheory.IsIso α.τ₂
    this : CategoryTheory.Mono α.τ₃
    ⊢ (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) f ⋯).Exact
  -/
  rw [ShortComplex.exact_iff_of_epi_of_isIso_of_mono α]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) …
    this✝¹ : CategoryTheory.Epi α.τ₁
    this✝ : CategoryTheory.IsIso α.τ₂
    this : CategoryTheory.Mono α.τ₃
    ⊢ (CategoryTheory.ShortComplex.mk (CategoryTheory.Limits.kernel.ι f) f ⋯).Exact
  -/
  apply ShortComplex.exact_of_f_is_kernel
  /-
    case hS
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    X Y : C
    f : Quiver.Hom X Y
    α : Quiver.Hom (CategoryTheory.ShortComplex.mk (CategoryTheory.Projective.d f) …
    this✝¹ : CategoryTheory.Epi α.τ₁
    this✝ : CategoryTheory.IsIso α.τ₂
    this : CategoryTheory.Mono α.τ₃
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Categor …
  -/
  apply kernelIsKernel
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `ProjectiveResolution.of`. -/
def ofComplex : ChainComplex C ℕ :=
  ChainComplex.mk' (Projective.over Z) (Projective.syzygies (Projective.π Z))
                                                                     /-
                                                                       C : Type u
                                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                                       inst✝¹ : CategoryTheory.Abelian C
                                                                       inst✝ : CategoryTheory.EnoughProjectives C
                                                                       Z X₀✝ X₁✝ : C
                                                                       f : Quiver.Hom X₁✝ X₀✝
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Projective.d f) f) 0
                                                                     -/
    (Projective.d (Projective.π Z)) (fun f => ⟨_, Projective.d f, by simp⟩)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma ofComplex_d_1_0 :
    (ofComplex Z).d 1 0 = d (Projective.π Z) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    Z : C
    ⊢ Eq ((CategoryTheory.ProjectiveResolution.ofComplex Z).d 1 0) (CategoryTheory …
  -/
  simp [ofComplex]
  /-
    🎉 no goals
  -/


lemma ofComplex_exactAt_succ (n : ℕ) :
    (ofComplex Z).ExactAt (n + 1) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    Z : C
    n : Nat
    ⊢ HomologicalComplex.ExactAt (CategoryTheory.ProjectiveResolution.ofComplex Z) …
  -/
  rw [HomologicalComplex.exactAt_iff' _ (n + 1 + 1) (n + 1) n (by simp) (by simp)]
  dsimp [ofComplex, HomologicalComplex.sc', HomologicalComplex.shortComplexFunctor',
      ChainComplex.mk', ChainComplex.mk]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    Z : C
    n : Nat
    ⊢ (CategoryTheory.ShortComplex.mk ((ChainComplex.of (fun n => (ChainComplex.mk …
  -/
  simp only [ChainComplex.of_d]
  -- TODO: this should just be apply exact_d_f so something is missing
  match n with
  | 0 => apply exact_d_f
  | n + 1 => apply exact_d_f


instance (n : ℕ) : Projective ((ofComplex Z).X n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.EnoughProjectives C
    Z : C
    n : Nat
    ⊢ CategoryTheory.Projective ((CategoryTheory.ProjectiveResolution.ofComplex Z) …
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
  obtain (_ | _ | _ | n) := n <;> apply Projective.projective_over
                                  /-
                                    🎉 no goals
                                  -/


/-- In any abelian category with enough projectives,
`ProjectiveResolution.of Z` constructs an projective resolution of the object `Z`.
-/
irreducible_def of : ProjectiveResolution Z where
  complex := ofComplex Z
  π := (ChainComplex.toSingle₀Equiv _ _).symm ⟨Projective.π Z, by
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Abelian C
            inst✝ : CategoryTheory.EnoughProjectives C
            Z : C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ProjectiveResolution …
          -/
          rw [ofComplex_d_1_0, assoc, kernel.condition, comp_zero]⟩
          /-
            🎉 no goals
          -/
  quasiIso := ⟨fun n => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.EnoughProjectives C
      Z : C
      n : Nat
      ⊢ QuasiIsoAt (((CategoryTheory.ProjectiveResolution.ofComplex Z).toSingle₀Equi …
    -/
    cases n
      /-
        case zero
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Abelian C
        inst✝ : CategoryTheory.EnoughProjectives C
        Z : C
        ⊢ QuasiIsoAt (((CategoryTheory.ProjectiveResolution.ofComplex Z).toSingle₀Equi …
      -/
    · rw [ChainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros']
        /-
          case zero
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.EnoughProjectives C
          Z : C
          ⊢ And (CategoryTheory.ShortComplex.mk ((HomologicalComplex.shortComplexFunctor …
        -/
      · dsimp
        refine (ShortComplex.exact_and_epi_g_iff_of_iso ?_).2
          ⟨exact_d_f (Projective.π Z), by dsimp; infer_instance⟩
        exact ShortComplex.isoMk (Iso.refl _) (Iso.refl _) (Iso.refl _)
          (by simp [ofComplex]) (by simp)
      /-
        case zero.hg₁
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Abelian C
        inst✝ : CategoryTheory.EnoughProjectives C
        Z : C
        ⊢ Eq ((HomologicalComplex.shortComplexFunctor' C (ComplexShape.down Nat) 1 0 0 …
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
        inst✝ : CategoryTheory.EnoughProjectives C
        Z : C
        n✝ : Nat
        ⊢ QuasiIsoAt (((CategoryTheory.ProjectiveResolution.ofComplex Z).toSingle₀Equi …
      -/
    · rw [quasiIsoAt_iff_exactAt']
        /-
          case succ
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.EnoughProjectives C
          Z : C
          n✝ : Nat
          ⊢ HomologicalComplex.ExactAt (CategoryTheory.ProjectiveResolution.ofComplex Z) …
        -/
      · apply ofComplex_exactAt_succ
        /-
          🎉 no goals
        -/
        /-
          case succ.hL
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.EnoughProjectives C
          Z : C
          n✝ : Nat
          ⊢ HomologicalComplex.ExactAt ((ChainComplex.single₀ C).obj Z) (HAdd.hAdd n✝ 1)
        -/
      · apply ChainComplex.exactAt_succ_single_obj⟩
        /-
          🎉 no goals
        -/


instance (priority := 100) (Z : C) : HasProjectiveResolution Z where out := ⟨of Z⟩


instance (priority := 100) : HasProjectiveResolutions C where out _ := inferInstance


