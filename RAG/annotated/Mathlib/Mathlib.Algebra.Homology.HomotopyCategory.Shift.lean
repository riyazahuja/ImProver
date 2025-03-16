/-- The shift functor by `n : ℤ` on `CochainComplex C ℤ` which sends a cochain
complex `K` to the complex which is `K.X (i + n)` in degree `i`, and which
multiplies the differentials by `(-1)^n`. -/
@[simps]
def shiftFunctor (n : ℤ) : CochainComplex C ℤ ⥤ CochainComplex C ℤ where
  obj K :=
    { X := fun i => K.X (i + n)
      d := fun _ _ => n.negOnePow • K.d _ _
      d_comp_d' := by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          ⊢ ∀ (i j k : Int), (ComplexShape.up Int).Rel i j → (ComplexShape.up Int).Rel j …
        -/
        intros
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i✝ j✝ k✝ : Int
          a✝¹ : (ComplexShape.up Int).Rel i✝ j✝
          a✝ : (ComplexShape.up Int).Rel j✝ k✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x x_1 => HSMul.hSMul n.negOnePo …
        -/
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i j : Int
          hij : Not ((ComplexShape.up Int).Rel i j)
          ⊢ Eq ((fun x x_1 => HSMul.hSMul n.negOnePow (K.d (HAdd.hAdd x n) (HAdd.hAdd x_ …
        -/
        simp only [Linear.comp_units_smul, Linear.units_smul_comp, d_comp_d, smul_zero]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i j : Int
          hij : Not ((ComplexShape.up Int).Rel i j)
          ⊢ Eq (HSMul.hSMul n.negOnePow (K.d (HAdd.hAdd i n) (HAdd.hAdd j n))) 0
        -/
        /-
          🎉 no goals
        -/
        /-
          case a
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i j : Int
          hij : Not ((ComplexShape.up Int).Rel i j)
          ⊢ Not ((ComplexShape.up Int).Rel (HAdd.hAdd i n) (HAdd.hAdd j n))
        -/
      shape := fun i j hij => by
        /-
          case a
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i j : Int
          hij : Not ((ComplexShape.up Int).Rel i j)
          hij' : (ComplexShape.up Int).Rel (HAdd.hAdd i n) (HAdd.hAdd j n)
          ⊢ False
        -/
        dsimp
        /-
          case a
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i j : Int
          hij : Not ((ComplexShape.up Int).Rel i j)
          hij' : (ComplexShape.up Int).Rel (HAdd.hAdd i n) (HAdd.hAdd j n)
          ⊢ (ComplexShape.up Int).Rel i j
        -/
        rw [K.shape, smul_zero]
        /-
          case a
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          K : CochainComplex C Int
          i j : Int
          hij : Not ((ComplexShape.up Int).Rel i j)
          hij' : Eq (HAdd.hAdd (HAdd.hAdd i n) 1) (HAdd.hAdd j n)
          ⊢ Eq (HAdd.hAdd i 1) j
        -/
        intro hij'
        /-
          🎉 no goals
        -/
        apply hij
        dsimp at hij' ⊢
        omega }
  map φ :=
    { f := fun _ => φ.f _
      comm' := by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          X✝ Y✝ : CochainComplex C Int
          φ : Quiver.Hom X✝ Y✝
          ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
        -/
        intros
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          X✝ Y✝ : CochainComplex C Int
          φ : Quiver.Hom X✝ Y✝
          i✝ j✝ : Int
          a✝ : (ComplexShape.up Int).Rel i✝ j✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => φ.f (HAdd.hAdd x n)) i✝) ( …
        -/
        dsimp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n : Int
          X✝ Y✝ : CochainComplex C Int
          φ : Quiver.Hom X✝ Y✝
          i✝ j✝ : Int
          a✝ : (ComplexShape.up Int).Rel i✝ j✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.f (HAdd.hAdd i✝ n)) (HSMul.hSMul n …
        -/
        simp only [Linear.comp_units_smul, Hom.comm, Linear.units_smul_comp] }
        /-
          🎉 no goals
        -/
               /-
                 C : Type u
                 inst✝³ : CategoryTheory.Category.{v, u} C
                 inst✝² : CategoryTheory.Preadditive C
                 D : Type u'
                 inst✝¹ : CategoryTheory.Category.{v', u'} D
                 inst✝ : CategoryTheory.Preadditive D
                 n : Int
                 ⊢ ∀ (X : CochainComplex C Int), Eq ({ obj := fun K => { X := fun i => K.X (HAd …
               -/
  map_id := by intros; rfl
                       /-
                         🎉 no goals
                       -/
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   inst✝² : CategoryTheory.Preadditive C
                   D : Type u'
                   inst✝¹ : CategoryTheory.Category.{v', u'} D
                   inst✝ : CategoryTheory.Preadditive D
                   n : Int
                   ⊢ ∀ {X Y Z : CochainComplex C Int} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z),  …
                 -/
  map_comp := by intros; rfl
                         /-
                           🎉 no goals
                         -/


instance (n : ℤ) : (shiftFunctor C n).Additive where


/-- The canonical isomorphism `((shiftFunctor C n).obj K).X i ≅ K.X m` when `m = i + n`. -/
@[simp]
def shiftFunctorObjXIso (K : CochainComplex C ℤ) (n i m : ℤ) (hm : m = i + n) :
    ((shiftFunctor C n).obj K).X i ≅ K.X m := K.XIsoOfEq hm.symm


/-- The shift functor by `n` on `CochainComplex C ℤ` identifies to the identity
functor when `n = 0`. -/
@[simps!]
def shiftFunctorZero' (n : ℤ) (h : n = 0) :
    shiftFunctor C n ≅ 𝟭 _ :=
  NatIso.ofComponents (fun K => Hom.isoOfComponents
                                              /-
                                                C : Type u
                                                inst✝³ : CategoryTheory.Category.{v, u} C
                                                inst✝² : CategoryTheory.Preadditive C
                                                D : Type u'
                                                inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                inst✝ : CategoryTheory.Preadditive D
                                                n : Int
                                                h : Eq n 0
                                                K : CochainComplex C Int
                                                i : Int
                                                ⊢ Eq i (HAdd.hAdd i n)
                                              -/
    (fun i => K.shiftFunctorObjXIso _ _ _ (by omega))
                                              /-
                                                🎉 no goals
                                              -/
                     /-
                       C : Type u
                       inst✝³ : CategoryTheory.Category.{v, u} C
                       inst✝² : CategoryTheory.Preadditive C
                       D : Type u'
                       inst✝¹ : CategoryTheory.Category.{v', u'} D
                       inst✝ : CategoryTheory.Preadditive D
                       n : Int
                       h : Eq n 0
                       K : CochainComplex C Int
                       x✝² x✝¹ : Int
                       x✝ : (ComplexShape.up Int).Rel x✝² x✝¹
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => K.shiftFunctorObjXIso n i  …
                     -/
                            /-
                              🎉 no goals
                            -/
    (fun _ _ _ => by dsimp; simp [h])) (fun _ ↦ by ext; dsimp; simp)
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The compatibility of the shift functors on `CochainComplex C ℤ` with respect
to the addition of integers. -/
@[simps!]
def shiftFunctorAdd' (n₁ n₂ n₁₂ : ℤ) (h : n₁ + n₂ = n₁₂) :
    shiftFunctor C n₁₂ ≅ shiftFunctor C n₁ ⋙ shiftFunctor C n₂ :=
  NatIso.ofComponents (fun K => Hom.isoOfComponents
                                              /-
                                                C : Type u
                                                inst✝³ : CategoryTheory.Category.{v, u} C
                                                inst✝² : CategoryTheory.Preadditive C
                                                D : Type u'
                                                inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                inst✝ : CategoryTheory.Preadditive D
                                                n₁ n₂ n₁₂ : Int
                                                h : Eq (HAdd.hAdd n₁ n₂) n₁₂
                                                K : CochainComplex C Int
                                                i : Int
                                                ⊢ Eq (HAdd.hAdd (HAdd.hAdd i n₂) n₁) (HAdd.hAdd i n₁₂)
                                              -/
    (fun i => K.shiftFunctorObjXIso _ _ _ (by omega))
                                              /-
                                                🎉 no goals
                                              -/
    (fun _ _ _ => by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.Preadditive D
        n₁ n₂ n₁₂ : Int
        h : Eq (HAdd.hAdd n₁ n₂) n₁₂
        K : CochainComplex C Int
        x✝² x✝¹ : Int
        x✝ : (ComplexShape.up Int).Rel x✝² x✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => K.shiftFunctorObjXIso n₁₂  …
      -/
      subst h
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        D : Type u'
        inst✝¹ : CategoryTheory.Category.{v', u'} D
        inst✝ : CategoryTheory.Preadditive D
        n₁ n₂ : Int
        K : CochainComplex C Int
        x✝² x✝¹ : Int
        x✝ : (ComplexShape.up Int).Rel x✝² x✝¹
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => K.shiftFunctorObjXIso (HAd …
      -/
      dsimp
      simp only [add_comm n₁ n₂, Int.negOnePow_add, Linear.units_smul_comp,
        Linear.comp_units_smul, d_comp_XIsoOfEq_hom, smul_smul, XIsoOfEq_hom_comp_d]))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n₁ n₂ n₁₂ : Int
          h : Eq (HAdd.hAdd n₁ n₂) n₁₂
          ⊢ ∀ {X Y : CochainComplex C Int} (f : Quiver.Hom X Y), Eq (CategoryTheory.Cate …
        -/
    (by intros; ext; dsimp; simp)
                            /-
                              🎉 no goals
                            -/


attribute [local simp] XIsoOfEq


instance : HasShift (CochainComplex C ℤ) ℤ := hasShiftMk _ _
  { F := shiftFunctor C
    zero := shiftFunctorZero' C _ rfl
    add := fun n₁ n₂ => shiftFunctorAdd' C n₁ n₂ _ rfl }


instance (n : ℤ) :
    (CategoryTheory.shiftFunctor (HomologicalComplex C (ComplexShape.up ℤ)) n).Additive :=
  (inferInstance : (CochainComplex.shiftFunctor C n).Additive)


@[simp]
lemma shiftFunctor_obj_X' (K : CochainComplex C ℤ) (n p : ℤ) :
    ((CategoryTheory.shiftFunctor (CochainComplex C ℤ) n).obj K).X p = K.X (p + n) := rfl


@[simp]
lemma shiftFunctor_map_f' {K L : CochainComplex C ℤ} (φ : K ⟶ L) (n p : ℤ) :
    ((CategoryTheory.shiftFunctor (CochainComplex C ℤ) n).map φ).f p = φ.f (p + n) := rfl


@[simp]
lemma shiftFunctor_obj_d' (K : CochainComplex C ℤ) (n i j : ℤ) :
    ((CategoryTheory.shiftFunctor (CochainComplex C ℤ) n).obj K).d i j =
      n.negOnePow • K.d _ _ := rfl


lemma shiftFunctorAdd_inv_app_f (K : CochainComplex C ℤ) (a b n : ℤ) :
    ((shiftFunctorAdd (CochainComplex C ℤ) a b).inv.app K).f n =
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝¹ : CategoryTheory.Category.{v', u'} D
                        inst✝ : CategoryTheory.Preadditive D
                        K : CochainComplex C Int
                        a b n : Int
                        ⊢ Eq (HAdd.hAdd (HAdd.hAdd n { as := b }.as) { as := a }.as) (HAdd.hAdd n { as …
                      -/
      (K.XIsoOfEq (by dsimp; rw [add_comm a, add_assoc])).hom := rfl
                             /-
                               🎉 no goals
                             -/


lemma shiftFunctorAdd_hom_app_f (K : CochainComplex C ℤ) (a b n : ℤ) :
    ((shiftFunctorAdd (CochainComplex C ℤ) a b).hom.app K).f n =
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝¹ : CategoryTheory.Category.{v', u'} D
                        inst✝ : CategoryTheory.Preadditive D
                        K : CochainComplex C Int
                        a b n : Int
                        ⊢ Eq (HAdd.hAdd n { as := HAdd.hAdd a b }.as) (HAdd.hAdd (HAdd.hAdd n { as :=  …
                      -/
      (K.XIsoOfEq (by dsimp; rw [add_comm a, add_assoc])).hom := by
                             /-
                               🎉 no goals
                             -/
  have : IsIso (((shiftFunctorAdd (CochainComplex C ℤ) a b).inv.app K).f n) := by
    rw [shiftFunctorAdd_inv_app_f]
    infer_instance
  rw [← cancel_mono (((shiftFunctorAdd (CochainComplex C ℤ) a b).inv.app K).f n),
    ← comp_f, Iso.hom_inv_id_app, id_f, shiftFunctorAdd_inv_app_f]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b n : Int
    this : CategoryTheory.IsIso (((CategoryTheory.shiftFunctorAdd (CochainComplex  …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.shiftFunctor (Cochain …
  -/
  simp only [XIsoOfEq, eqToIso.hom, eqToHom_trans, eqToHom_refl]
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_inv_app_f' (K : CochainComplex C ℤ) (a b ab : ℤ) (h : a + b = ab) (n : ℤ) :
    ((CategoryTheory.shiftFunctorAdd' (CochainComplex C ℤ) a b ab h).inv.app K).f n =
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝¹ : CategoryTheory.Category.{v', u'} D
                        inst✝ : CategoryTheory.Preadditive D
                        K : CochainComplex C Int
                        a b ab : Int
                        h : Eq (HAdd.hAdd a b) ab
                        n : Int
                        ⊢ Eq (HAdd.hAdd (HAdd.hAdd n { as := b }.as) { as := a }.as) (HAdd.hAdd n { as …
                      -/
      (K.XIsoOfEq (by dsimp; rw [← h, add_assoc, add_comm a])).hom := by
                             /-
                               🎉 no goals
                             -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b ab : Int
    h : Eq (HAdd.hAdd a b) ab
    n : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b ab h).inv.a …
  -/
  subst h
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b n : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b (HAdd.hAdd  …
  -/
  rw [shiftFunctorAdd'_eq_shiftFunctorAdd, shiftFunctorAdd_inv_app_f]
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_hom_app_f' (K : CochainComplex C ℤ) (a b ab : ℤ) (h : a + b = ab) (n : ℤ) :
    ((CategoryTheory.shiftFunctorAdd' (CochainComplex C ℤ) a b ab h).hom.app K).f n =
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝¹ : CategoryTheory.Category.{v', u'} D
                        inst✝ : CategoryTheory.Preadditive D
                        K : CochainComplex C Int
                        a b ab : Int
                        h : Eq (HAdd.hAdd a b) ab
                        n : Int
                        ⊢ Eq (HAdd.hAdd n { as := ab }.as) (HAdd.hAdd (HAdd.hAdd n { as := b }.as) { a …
                      -/
      (K.XIsoOfEq (by dsimp; rw [← h, add_assoc, add_comm a])).hom := by
                             /-
                               🎉 no goals
                             -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b ab : Int
    h : Eq (HAdd.hAdd a b) ab
    n : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b ab h).hom.a …
  -/
  subst h
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b n : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b (HAdd.hAdd  …
  -/
  rw [shiftFunctorAdd'_eq_shiftFunctorAdd, shiftFunctorAdd_hom_app_f]
  /-
    🎉 no goals
  -/


lemma shiftFunctorZero_inv_app_f (K : CochainComplex C ℤ) (n : ℤ) :
    ((CategoryTheory.shiftFunctorZero (CochainComplex C ℤ) ℤ).inv.app K).f n =
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝¹ : CategoryTheory.Category.{v', u'} D
                        inst✝ : CategoryTheory.Preadditive D
                        K : CochainComplex C Int
                        n : Int
                        ⊢ Eq n (HAdd.hAdd n { as := 0 }.as)
                      -/
      (K.XIsoOfEq (by dsimp; rw [add_zero])).hom := rfl
                             /-
                               🎉 no goals
                             -/


lemma shiftFunctorZero_hom_app_f (K : CochainComplex C ℤ) (n : ℤ) :
    ((CategoryTheory.shiftFunctorZero (CochainComplex C ℤ) ℤ).hom.app K).f n =
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝¹ : CategoryTheory.Category.{v', u'} D
                        inst✝ : CategoryTheory.Preadditive D
                        K : CochainComplex C Int
                        n : Int
                        ⊢ Eq (HAdd.hAdd n { as := 0 }.as) n
                      -/
      (K.XIsoOfEq (by dsimp; rw [add_zero])).hom := by
                             /-
                               🎉 no goals
                             -/
  have : IsIso (((shiftFunctorZero (CochainComplex C ℤ) ℤ).inv.app K).f n) := by
    rw [shiftFunctorZero_inv_app_f]
    infer_instance
  rw [← cancel_mono (((shiftFunctorZero (CochainComplex C ℤ) ℤ).inv.app K).f n), ← comp_f,
    Iso.hom_inv_id_app, id_f, shiftFunctorZero_inv_app_f]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    n : Int
    this : CategoryTheory.IsIso (((CategoryTheory.shiftFunctorZero (CochainComplex …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.shiftFunctor (Cochain …
  -/
  simp only [XIsoOfEq, eqToIso.hom, eqToHom_trans, eqToHom_refl]
  /-
    🎉 no goals
  -/


lemma XIsoOfEq_shift (K : CochainComplex C ℤ) (n : ℤ) {p q : ℤ} (hpq : p = q) :
                                                            /-
                                                              C : Type u
                                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                                              inst✝² : CategoryTheory.Preadditive C
                                                              D : Type u'
                                                              inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                              inst✝ : CategoryTheory.Preadditive D
                                                              K : CochainComplex C Int
                                                              n p q : Int
                                                              hpq : Eq p q
                                                              ⊢ Eq (HAdd.hAdd p n) (HAdd.hAdd q n)
                                                            -/
    (K⟦n⟧).XIsoOfEq hpq = K.XIsoOfEq (show p + n = q + n by rw [hpq]) := rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma shiftFunctorAdd'_eq (a b c : ℤ) (h : a + b = c) :
    CategoryTheory.shiftFunctorAdd' (CochainComplex C ℤ) a b c h =
      shiftFunctorAdd' C a b c h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    a b c : Int
    h : Eq (HAdd.hAdd a b) c
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b c h) (Cochain …
  -/
  ext
  /-
    case w.w.h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    a b c : Int
    h : Eq (HAdd.hAdd a b) c
    x✝ : CochainComplex C Int
    i✝ : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b c h).hom.ap …
  -/
  simp only [shiftFunctorAdd'_hom_app_f', XIsoOfEq, eqToIso.hom, shiftFunctorAdd'_hom_app_f]
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd_eq (a b : ℤ) :
    CategoryTheory.shiftFunctorAdd (CochainComplex C ℤ) a b = shiftFunctorAdd' C a b _ rfl := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    a b : Int
    ⊢ Eq (CategoryTheory.shiftFunctorAdd (CochainComplex C Int) a b) (CochainCompl …
  -/
  rw [← CategoryTheory.shiftFunctorAdd'_eq_shiftFunctorAdd, shiftFunctorAdd'_eq]
  /-
    🎉 no goals
  -/


lemma shiftFunctorZero_eq :
    CategoryTheory.shiftFunctorZero (CochainComplex C ℤ) ℤ = shiftFunctorZero' C 0 rfl := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    ⊢ Eq (CategoryTheory.shiftFunctorZero (CochainComplex C Int) Int) (CochainComp …
  -/
  ext
  /-
    case w.w.h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    x✝ : CochainComplex C Int
    i✝ : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorZero (CochainComplex C Int) Int).hom.app x✝ …
  -/
  rw [shiftFunctorZero_hom_app_f, shiftFunctorZero'_hom_app_f]
  /-
    🎉 no goals
  -/


lemma shiftFunctorComm_hom_app_f (K : CochainComplex C ℤ) (a b p : ℤ) :
    ((shiftFunctorComm (CochainComplex C ℤ) a b).hom.app K).f p =
      (K.XIsoOfEq (show p + b + a = p + a + b
           /-
             C : Type u
             inst✝³ : CategoryTheory.Category.{v, u} C
             inst✝² : CategoryTheory.Preadditive C
             D : Type u'
             inst✝¹ : CategoryTheory.Category.{v', u'} D
             inst✝ : CategoryTheory.Preadditive D
             K : CochainComplex C Int
             a b p : Int
             ⊢ Eq (HAdd.hAdd (HAdd.hAdd p b) a) (HAdd.hAdd (HAdd.hAdd p a) b)
           -/
        by rw [add_assoc, add_comm b, add_assoc])).hom := by
           /-
             🎉 no goals
           -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b p : Int
    ⊢ Eq (((CategoryTheory.shiftFunctorComm (CochainComplex C Int) a b).hom.app K) …
  -/
  rw [shiftFunctorComm_eq _ _ _ _ rfl]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b p : Int
    ⊢ Eq ((((CategoryTheory.shiftFunctorAdd' (CochainComplex C Int) a b (HAdd.hAdd …
  -/
  dsimp
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b p : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.shiftFunctorAdd' (C …
  -/
  rw [shiftFunctorAdd'_inv_app_f', shiftFunctorAdd'_hom_app_f']
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    K : CochainComplex C Int
    a b p : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.XIsoOfEq K ⋯).hom …
  -/
  simp only [XIsoOfEq, eqToIso.hom, eqToHom_trans]
  /-
    🎉 no goals
  -/


/-- Shifting cochain complexes by `n` and evaluating in a degree `i` identifies
to the evaluation in degree `i'` when `n + i = i'`. -/
@[simps!]
def shiftEval (n i i' : ℤ) (hi : n + i = i') :
    (CategoryTheory.shiftFunctor (CochainComplex C ℤ) n) ⋙
      HomologicalComplex.eval C (ComplexShape.up ℤ) i ≅
      HomologicalComplex.eval C (ComplexShape.up ℤ) i' :=
                                               /-
                                                 C : Type u
                                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                                 inst✝² : CategoryTheory.Preadditive C
                                                 D : Type u'
                                                 inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                 inst✝ : CategoryTheory.Preadditive D
                                                 n i i' : Int
                                                 hi : Eq (HAdd.hAdd n i) i'
                                                 K : CochainComplex C Int
                                                 ⊢ Eq (HAdd.hAdd i { as := n }.as) i'
                                               -/
  NatIso.ofComponents (fun K => K.XIsoOfEq (by dsimp; rw [← hi, add_comm i]))
                                                      /-
                                                        🎉 no goals
                                                      -/
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          D : Type u'
          inst✝¹ : CategoryTheory.Category.{v', u'} D
          inst✝ : CategoryTheory.Preadditive D
          n i i' : Int
          hi : Eq (HAdd.hAdd n i) i'
          ⊢ ∀ {X Y : CochainComplex C Int} (f : Quiver.Hom X Y), Eq (CategoryTheory.Cate …
        -/
    (by intros; dsimp; simp)
                       /-
                         🎉 no goals
                       -/


/-- The commutation with the shift isomorphism for the functor on cochain complexes
induced by an additive functor between preadditive categories. -/
@[simps!]
def mapCochainComplexShiftIso (n : ℤ) :
    shiftFunctor _ n ⋙ F.mapHomologicalComplex (ComplexShape.up ℤ) ≅
      F.mapHomologicalComplex (ComplexShape.up ℤ) ⋙ shiftFunctor _ n :=
  NatIso.ofComponents (fun K => HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _)
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Preadditive C
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          inst✝¹ : CategoryTheory.Preadditive D
          F : CategoryTheory.Functor C D
          inst✝ : F.Additive
          n : Int
          K : HomologicalComplex C (ComplexShape.up Int)
          ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
        -/
               /-
                 🎉 no goals
               -/
    (by dsimp; simp)) (fun _ => by ext; dsimp; rw [id_comp, comp_id])
                                               /-
                                                 🎉 no goals
                                               -/


instance commShiftMapCochainComplex :
    (F.mapHomologicalComplex (ComplexShape.up ℤ)).CommShift ℤ where
  iso := F.mapCochainComplexShiftIso
  zero := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (F.mapCochainComplexShiftIso 0) (CategoryTheory.Functor.CommShift.isoZero …
    -/
    ext
    /-
      case w.w.h.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      x✝ : HomologicalComplex C (ComplexShape.up Int)
      i✝ : Int
      ⊢ Eq (((F.mapCochainComplexShiftIso 0).hom.app x✝).f i✝) (((CategoryTheory.Fun …
    -/
    rw [CommShift.isoZero_hom_app]
    /-
      case w.w.h.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      x✝ : HomologicalComplex C (ComplexShape.up Int)
      i✝ : Int
      ⊢ Eq (((F.mapCochainComplexShiftIso 0).hom.app x✝).f i✝) ((CategoryTheory.Cate …
    -/
    dsimp
    simp only [mapCochainComplexShiftIso_hom_app_f, CochainComplex.shiftFunctorZero_inv_app_f,
       CochainComplex.shiftFunctorZero_hom_app_f, HomologicalComplex.XIsoOfEq, eqToIso,
       eqToHom_map, eqToHom_trans, eqToHom_refl]
  add := fun a b => by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      a b : Int
      ⊢ Eq (F.mapCochainComplexShiftIso (HAdd.hAdd a b)) (CategoryTheory.Functor.Com …
    -/
    ext
    /-
      case w.w.h.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      a b : Int
      x✝ : HomologicalComplex C (ComplexShape.up Int)
      i✝ : Int
      ⊢ Eq (((F.mapCochainComplexShiftIso (HAdd.hAdd a b)).hom.app x✝).f i✝) (((Cate …
    -/
    rw [CommShift.isoAdd_hom_app]
    /-
      case w.w.h.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      a b : Int
      x✝ : HomologicalComplex C (ComplexShape.up Int)
      i✝ : Int
      ⊢ Eq (((F.mapCochainComplexShiftIso (HAdd.hAdd a b)).hom.app x✝).f i✝) ((Categ …
    -/
    dsimp
    /-
      case w.w.h.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      a b : Int
      x✝ : HomologicalComplex C (ComplexShape.up Int)
      i✝ : Int
      ⊢ Eq (CategoryTheory.CategoryStruct.id (F.obj (x✝.X (HAdd.hAdd i✝ (HAdd.hAdd a …
    -/
    rw [id_comp, id_comp]
    simp only [CochainComplex.shiftFunctorAdd_hom_app_f,
      CochainComplex.shiftFunctorAdd_inv_app_f, HomologicalComplex.XIsoOfEq, eqToIso,
      eqToHom_map, eqToHom_trans, eqToHom_refl]


lemma mapHomologicalComplex_commShiftIso_eq (n : ℤ) :
    (F.mapHomologicalComplex (ComplexShape.up ℤ)).commShiftIso n =
      F.mapCochainComplexShiftIso n := rfl


@[simp]
lemma mapHomologicalComplex_commShiftIso_hom_app_f (K : CochainComplex C ℤ) (n i : ℤ) :
    (((F.mapHomologicalComplex (ComplexShape.up ℤ)).commShiftIso n).hom.app K).f i = 𝟙 _ := rfl


@[simp]
lemma mapHomologicalComplex_commShiftIso_inv_app_f (K : CochainComplex C ℤ) (n i : ℤ) :
    (((F.mapHomologicalComplex (ComplexShape.up ℤ)).commShiftIso n).inv.app K).f i = 𝟙 _ := rfl


/-- If `h : Homotopy φ₁ φ₂` and `n : ℤ`, this is the induced homotopy
between `φ₁⟦n⟧'` and `φ₂⟦n⟧'`. -/
def shift {K L : CochainComplex C ℤ} {φ₁ φ₂ : K ⟶ L} (h : Homotopy φ₁ φ₂) (n : ℤ) :
    Homotopy (φ₁⟦n⟧') (φ₂⟦n⟧') where
  hom _ _ := n.negOnePow • h.hom _ _
  zero i j hij := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Preadditive D
      K L : CochainComplex C Int
      φ₁ φ₂ : Quiver.Hom K L
      h : Homotopy φ₁ φ₂
      n i j : Int
      hij : Not ((ComplexShape.up Int).Rel j i)
      ⊢ Eq ((fun x x_1 => HSMul.hSMul n.negOnePow (h.hom (HAdd.hAdd x { as := n }.as …
    -/
    dsimp
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Preadditive D
      K L : CochainComplex C Int
      φ₁ φ₂ : Quiver.Hom K L
      h : Homotopy φ₁ φ₂
      n i j : Int
      hij : Not ((ComplexShape.up Int).Rel j i)
      ⊢ Eq (HSMul.hSMul n.negOnePow (h.hom (HAdd.hAdd i n) (HAdd.hAdd j n))) 0
    -/
    rw [h.zero, smul_zero]
    /-
      case a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Preadditive D
      K L : CochainComplex C Int
      φ₁ φ₂ : Quiver.Hom K L
      h : Homotopy φ₁ φ₂
      n i j : Int
      hij : Not ((ComplexShape.up Int).Rel j i)
      ⊢ Not ((ComplexShape.up Int).Rel (HAdd.hAdd j n) (HAdd.hAdd i n))
    -/
    intro hij'
    /-
      case a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Preadditive D
      K L : CochainComplex C Int
      φ₁ φ₂ : Quiver.Hom K L
      h : Homotopy φ₁ φ₂
      n i j : Int
      hij : Not ((ComplexShape.up Int).Rel j i)
      hij' : (ComplexShape.up Int).Rel (HAdd.hAdd j n) (HAdd.hAdd i n)
      ⊢ False
    -/
    dsimp at hij hij'
    /-
      case a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Preadditive D
      K L : CochainComplex C Int
      φ₁ φ₂ : Quiver.Hom K L
      h : Homotopy φ₁ φ₂
      n i j : Int
      hij : Not (Eq (HAdd.hAdd j 1) i)
      hij' : Eq (HAdd.hAdd (HAdd.hAdd j n) 1) (HAdd.hAdd i n)
      ⊢ False
    -/
    omega
    /-
      🎉 no goals
    -/
  comm := fun i => by
    rw [dNext_eq _ (show (ComplexShape.up ℤ).Rel i (i + 1) by simp),
      prevD_eq _ (show (ComplexShape.up ℤ).Rel (i - 1) i by simp)]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Preadditive D
      K L : CochainComplex C Int
      φ₁ φ₂ : Quiver.Hom K L
      h : Homotopy φ₁ φ₂
      n i : Int
      ⊢ Eq (((CategoryTheory.shiftFunctor (HomologicalComplex C (ComplexShape.up Int …
    -/
    dsimp
    simpa only [Linear.units_smul_comp, Linear.comp_units_smul, smul_smul,
      Int.units_mul_self, one_smul,
      dNext_eq _ (show (ComplexShape.up ℤ).Rel (i + n) (i + 1 + n) by dsimp; omega),
      prevD_eq _ (show (ComplexShape.up ℤ).Rel (i - 1 + n) (i + n) by dsimp; omega)]
        using h.comm (i + n)


instance : (homotopic C (ComplexShape.up ℤ)).IsCompatibleWithShift ℤ :=
  ⟨fun n _ _ _ _ ⟨h⟩ => ⟨h.shift n⟩⟩


noncomputable instance hasShift :
    HasShift (HomotopyCategory C (ComplexShape.up ℤ)) ℤ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Preadditive D
    ⊢ CategoryTheory.HasShift (HomotopyCategory C (ComplexShape.up Int)) Int
  -/
  dsimp only [HomotopyCategory]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Preadditive D
    ⊢ CategoryTheory.HasShift (CategoryTheory.Quotient (homotopic C (ComplexShape. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance commShiftQuotient :
    (HomotopyCategory.quotient C (ComplexShape.up ℤ)).CommShift ℤ :=
  Quotient.functor_commShift (homotopic C (ComplexShape.up ℤ)) ℤ


instance (n : ℤ) : (shiftFunctor (HomotopyCategory C (ComplexShape.up ℤ)) n).Additive := by
  have : ((quotient C (ComplexShape.up ℤ) ⋙ shiftFunctor _ n)).Additive :=
    Functor.additive_of_iso ((quotient C (ComplexShape.up ℤ)).commShiftIso n)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Preadditive D
    n : Int
    this : ((HomotopyCategory.quotient C (ComplexShape.up Int)).comp (CategoryTheo …
    ⊢ (CategoryTheory.shiftFunctor (HomotopyCategory C (ComplexShape.up Int)) n).A …
  -/
  apply Functor.additive_of_full_essSurj_comp (quotient _ _ )
  /-
    🎉 no goals
  -/


noncomputable instance : (F.mapHomotopyCategory (ComplexShape.up ℤ)).CommShift ℤ :=
  Quotient.liftCommShift _ _ _ _


instance : NatTrans.CommShift (F.mapHomotopyCategoryFactors (ComplexShape.up ℤ)).hom ℤ :=
  Quotient.liftCommShift_compatibility _ _ _ _


