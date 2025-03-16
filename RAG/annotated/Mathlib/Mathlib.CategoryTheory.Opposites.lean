theorem Quiver.Hom.op_inj {X Y : C} :
    Function.Injective (Quiver.Hom.op : (X ⟶ Y) → (Opposite.op Y ⟶ Opposite.op X)) := fun _ _ H =>
  congr_arg Quiver.Hom.unop H


theorem Quiver.Hom.unop_inj {X Y : Cᵒᵖ} :
    Function.Injective (Quiver.Hom.unop : (X ⟶ Y) → (Opposite.unop Y ⟶ Opposite.unop X)) :=
  fun _ _ H => congr_arg Quiver.Hom.op H


@[simp]
theorem Quiver.Hom.unop_op {X Y : C} (f : X ⟶ Y) : f.op.unop = f :=
  rfl


@[simp]
theorem Quiver.Hom.unop_op' {X Y : Cᵒᵖ} {x} :
    @Quiver.Hom.unop C _ X Y no_index (Opposite.op (unop := x)) = x := rfl


@[simp]
theorem Quiver.Hom.op_unop {X Y : Cᵒᵖ} (f : X ⟶ Y) : f.unop.op = f :=
  rfl


@[simp] theorem Quiver.Hom.unop_mk {X Y : Cᵒᵖ} (f : X ⟶ Y) : Quiver.Hom.unop {unop := f} = f := rfl


/-- The opposite category.

See <https://stacks.math.columbia.edu/tag/001M>.
-/
instance Category.opposite : Category.{v₁} Cᵒᵖ where
  comp f g := (g.unop ≫ f.unop).op
  id X := (𝟙 (unop X)).op


@[simp, reassoc]
theorem op_comp {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} : (f ≫ g).op = g.op ≫ f.op :=
  rfl


@[simp]
theorem op_id {X : C} : (𝟙 X).op = 𝟙 (op X) :=
  rfl


@[simp, reassoc]
theorem unop_comp {X Y Z : Cᵒᵖ} {f : X ⟶ Y} {g : Y ⟶ Z} : (f ≫ g).unop = g.unop ≫ f.unop :=
  rfl


@[simp]
theorem unop_id {X : Cᵒᵖ} : (𝟙 X).unop = 𝟙 (unop X) :=
  rfl


@[simp]
theorem unop_id_op {X : C} : (𝟙 (op X)).unop = 𝟙 X :=
  rfl


@[simp]
theorem op_id_unop {X : Cᵒᵖ} : (𝟙 (unop X)).op = 𝟙 X :=
  rfl


/-- The functor from the double-opposite of a category to the underlying category. -/
@[simps]
def unopUnop : Cᵒᵖᵒᵖ ⥤ C where
  obj X := unop (unop X)
  map f := f.unop.unop


/-- The functor from a category to its double-opposite. -/
@[simps]
def opOp : C ⥤ Cᵒᵖᵒᵖ where
  obj X := op (op X)
  map f := f.op.op


/-- The double opposite category is equivalent to the original. -/
@[simps]
def opOpEquivalence : Cᵒᵖᵒᵖ ≌ C where
  functor := unopUnop C
  inverse := opOp C
  unitIso := Iso.refl (𝟭 Cᵒᵖᵒᵖ)
  counitIso := Iso.refl (opOp C ⋙ unopUnop C)


/-- If `f` is an isomorphism, so is `f.op` -/
instance isIso_op {X Y : C} (f : X ⟶ Y) [IsIso f] : IsIso f.op :=
                                         /-
                                           C : Type u₁
                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                           X Y : C
                                           f : Quiver.Hom X Y
                                           inst✝ : CategoryTheory.IsIso f
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f.op (CategoryTheory.inv f).op).unop  …
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  ⟨⟨(inv f).op, ⟨Quiver.Hom.unop_inj (by aesop_cat), Quiver.Hom.unop_inj (by aesop_cat)⟩⟩⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- If `f.op` is an isomorphism `f` must be too.
(This cannot be an instance as it would immediately loop!)
-/
theorem isIso_of_op {X Y : C} (f : X ⟶ Y) [IsIso f.op] : IsIso f :=
                                            /-
                                              C : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                              X Y : C
                                              f : Quiver.Hom X Y
                                              inst✝ : CategoryTheory.IsIso f.op
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.inv f.op).unop).op  …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  ⟨⟨(inv f.op).unop, ⟨Quiver.Hom.op_inj (by simp), Quiver.Hom.op_inj (by simp)⟩⟩⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem isIso_op_iff {X Y : C} (f : X ⟶ Y) : IsIso f.op ↔ IsIso f :=
  ⟨fun _ => isIso_of_op _, fun _ => inferInstance⟩


theorem isIso_unop_iff {X Y : Cᵒᵖ} (f : X ⟶ Y) : IsIso f.unop ↔ IsIso f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f.unop) (CategoryTheory.IsIso f)
  -/
  rw [← isIso_op_iff f.unop, Quiver.Hom.op_unop]
  /-
    🎉 no goals
  -/


instance isIso_unop {X Y : Cᵒᵖ} (f : X ⟶ Y) [IsIso f] : IsIso f.unop :=
  (isIso_unop_iff _).2 inferInstance


@[simp]
theorem op_inv {X Y : C} (f : X ⟶ Y) [IsIso f] : (inv f).op = inv f.op := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.inv f).op (CategoryTheory.inv f.op)
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.op (CategoryTheory.inv f).op) (Cate …
  -/
  rw [← op_comp, IsIso.inv_hom_id, op_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem unop_inv {X Y : Cᵒᵖ} (f : X ⟶ Y) [IsIso f] : (inv f).unop = inv f.unop := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.inv f).unop (CategoryTheory.inv f.unop)
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop (CategoryTheory.inv f).unop) ( …
  -/
  rw [← unop_comp, IsIso.inv_hom_id, unop_id]
  /-
    🎉 no goals
  -/


/-- The opposite of a functor, i.e. considering a functor `F : C ⥤ D` as a functor `Cᵒᵖ ⥤ Dᵒᵖ`.
In informal mathematics no distinction is made between these. -/
@[simps]
protected def op (F : C ⥤ D) : Cᵒᵖ ⥤ Dᵒᵖ where
  obj X := op (F.obj (unop X))
  map f := (F.map f.unop).op


/-- Given a functor `F : Cᵒᵖ ⥤ Dᵒᵖ` we can take the "unopposite" functor `F : C ⥤ D`.
In informal mathematics no distinction is made between these.
-/
@[simps]
protected def unop (F : Cᵒᵖ ⥤ Dᵒᵖ) : C ⥤ D where
  obj X := unop (F.obj (op X))
  map f := (F.map f.op).unop


/-- The isomorphism between `F.op.unop` and `F`. -/
@[simps!]
def opUnopIso (F : C ⥤ D) : F.op.unop ≅ F :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The isomorphism between `F.unop.op` and `F`. -/
@[simps!]
def unopOpIso (F : Cᵒᵖ ⥤ Dᵒᵖ) : F.unop.op ≅ F :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor (Opposite C) (Opposite D)
    ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- Taking the opposite of a functor is functorial.
-/
@[simps]
def opHom : (C ⥤ D)ᵒᵖ ⥤ Cᵒᵖ ⥤ Dᵒᵖ where
  obj F := (unop F).op
  map α :=
    { app := fun X => (α.unop.app (unop X)).op
      naturality := fun _ _ f => Quiver.Hom.unop_inj (α.unop.naturality f.unop).symm }


/-- Take the "unopposite" of a functor is functorial.
-/
@[simps]
def opInv : (Cᵒᵖ ⥤ Dᵒᵖ) ⥤ (C ⥤ D)ᵒᵖ where
  obj F := op F.unop
  map α :=
    Quiver.Hom.op
      { app := fun X => (α.app (op X)).unop
        naturality := fun _ _ f => Quiver.Hom.op_inj <| (α.naturality f.op).symm }


/--
Another variant of the opposite of functor, turning a functor `C ⥤ Dᵒᵖ` into a functor `Cᵒᵖ ⥤ D`.
In informal mathematics no distinction is made.
-/
@[simps]
protected def leftOp (F : C ⥤ Dᵒᵖ) : Cᵒᵖ ⥤ D where
  obj X := unop (F.obj (unop X))
  map f := (F.map f.unop).unop


/--
Another variant of the opposite of functor, turning a functor `Cᵒᵖ ⥤ D` into a functor `C ⥤ Dᵒᵖ`.
In informal mathematics no distinction is made.
-/
@[simps]
protected def rightOp (F : Cᵒᵖ ⥤ D) : C ⥤ Dᵒᵖ where
  obj X := op (F.obj (op X))
  map f := (F.map f.op).op


lemma rightOp_map_unop {F : Cᵒᵖ ⥤ D} {X Y} (f : X ⟶ Y) :
    (F.rightOp.map f).unop = F.map f.op := rfl


instance {F : C ⥤ D} [Full F] : Full F.op where
                                                  /-
                                                    C : Type u₁
                                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                    D : Type u₂
                                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                    F : CategoryTheory.Functor C D
                                                    inst✝ : F.Full
                                                    X✝ Y✝ : Opposite C
                                                    f : Quiver.Hom (F.op.obj X✝) (F.op.obj Y✝)
                                                    ⊢ Eq (F.op.map (F.preimage f.unop).op) f
                                                  -/
  map_surjective f := ⟨(F.preimage f.unop).op, by simp⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


instance {F : C ⥤ D} [Faithful F] : Faithful F.op where
                                               /-
                                                 C : Type u₁
                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                 D : Type u₂
                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                 F : CategoryTheory.Functor C D
                                                 inst✝ : F.Faithful
                                                 X✝ Y✝ : Opposite C
                                                 a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                                 h : Eq (F.op.map a₁✝) (F.op.map a₂✝)
                                                 ⊢ Eq a₁✝.unop a₂✝.unop
                                               -/
  map_injective h := Quiver.Hom.unop_inj <| by simpa using map_injective F (Quiver.Hom.op_inj h)
                                               /-
                                                 🎉 no goals
                                               -/


/-- If F is faithful then the right_op of F is also faithful. -/
instance rightOp_faithful {F : Cᵒᵖ ⥤ D} [Faithful F] : Faithful F.rightOp where
  map_injective h := Quiver.Hom.op_inj (map_injective F (Quiver.Hom.op_inj h))


/-- If F is faithful then the left_op of F is also faithful. -/
instance leftOp_faithful {F : C ⥤ Dᵒᵖ} [Faithful F] : Faithful F.leftOp where
  map_injective h := Quiver.Hom.unop_inj (map_injective F (Quiver.Hom.unop_inj h))


instance rightOp_full {F : Cᵒᵖ ⥤ D} [Full F] : Full F.rightOp where
                                                    /-
                                                      C : Type u₁
                                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                      D : Type u₂
                                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                      F : CategoryTheory.Functor (Opposite C) D
                                                      inst✝ : F.Full
                                                      X✝ Y✝ : C
                                                      f : Quiver.Hom (F.rightOp.obj X✝) (F.rightOp.obj Y✝)
                                                      ⊢ Eq (F.rightOp.map (F.preimage f.unop).unop) f
                                                    -/
  map_surjective f := ⟨(F.preimage f.unop).unop, by simp⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


instance leftOp_full {F : C ⥤ Dᵒᵖ} [Full F] : Full F.leftOp where
                                                /-
                                                  C : Type u₁
                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                  D : Type u₂
                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                  F : CategoryTheory.Functor C (Opposite D)
                                                  inst✝ : F.Full
                                                  X✝ Y✝ : Opposite C
                                                  f : Quiver.Hom (F.leftOp.obj X✝) (F.leftOp.obj Y✝)
                                                  ⊢ Eq (F.leftOp.map (F.preimage f.op).op) f
                                                -/
  map_surjective f := ⟨(F.preimage f.op).op, by simp⟩
                                                /-
                                                  🎉 no goals
                                                -/



/-- The isomorphism between `F.leftOp.rightOp` and `F`. -/
@[simps!]
def leftOpRightOpIso (F : C ⥤ Dᵒᵖ) : F.leftOp.rightOp ≅ F :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C (Opposite D)
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The isomorphism between `F.rightOp.leftOp` and `F`. -/
@[simps!]
def rightOpLeftOpIso (F : Cᵒᵖ ⥤ D) : F.rightOp.leftOp ≅ F :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor (Opposite C) D
    ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- Whenever possible, it is advisable to use the isomorphism `rightOpLeftOpIso`
instead of this equality of functors. -/
theorem rightOp_leftOp_eq (F : Cᵒᵖ ⥤ D) : F.rightOp.leftOp = F := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq F.rightOp.leftOp F
  -/
  cases F
  /-
    case mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    toPrefunctor✝ : Prefunctor (Opposite C) D
    map_id✝ : ∀ (X : Opposite C), Eq (toPrefunctor✝.map (CategoryTheory.CategorySt …
    map_comp✝ : ∀ {X Y Z : Opposite C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z),  …
    ⊢ Eq { toPrefunctor := toPrefunctor✝, map_id := map_id✝, map_comp := map_comp✝ …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The opposite of a natural transformation. -/
@[simps]
protected def op (α : F ⟶ G) : G.op ⟶ F.op where
  app X := (α.app (unop X)).op
                                              /-
                                                C : Type u₁
                                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                D : Type u₂
                                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                F G : CategoryTheory.Functor C D
                                                α : Quiver.Hom F G
                                                X Y : Opposite C
                                                f : Quiver.Hom X Y
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.op.map f) ((fun X => (α.app (Oppos …
                                              -/
  naturality X Y f := Quiver.Hom.unop_inj (by simp)
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem op_id (F : C ⥤ D) : NatTrans.op (𝟙 F) = 𝟙 F.op :=
  rfl


/-- The "unopposite" of a natural transformation. -/
@[simps]
protected def unop {F G : Cᵒᵖ ⥤ Dᵒᵖ} (α : F ⟶ G) : G.unop ⟶ F.unop where
  app X := (α.app (op X)).unop
                                            /-
                                              C : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                              D : Type u₂
                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                              F✝ G✝ : CategoryTheory.Functor C D
                                              F G : CategoryTheory.Functor (Opposite C) (Opposite D)
                                              α : Quiver.Hom F G
                                              X Y : C
                                              f : Quiver.Hom X Y
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.unop.map f) ((fun X => (α.app { un …
                                            -/
  naturality X Y f := Quiver.Hom.op_inj (by simp)
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem unop_id (F : Cᵒᵖ ⥤ Dᵒᵖ) : NatTrans.unop (𝟙 F) = 𝟙 F.unop :=
  rfl


/-- Given a natural transformation `α : F.op ⟶ G.op`,
we can take the "unopposite" of each component obtaining a natural transformation `G ⟶ F`.
-/
@[simps]
protected def removeOp (α : F.op ⟶ G.op) : G ⟶ F where
  app X := (α.app (op X)).unop
  naturality X Y f :=
                            /-
                              C : Type u₁
                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                              D : Type u₂
                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                              F G : CategoryTheory.Functor C D
                              α : Quiver.Hom F.op G.op
                              X Y : C
                              f : Quiver.Hom X Y
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (α.app { unop := …
                            -/
    Quiver.Hom.op_inj <| by simpa only [Functor.op_map] using (α.naturality f.op).symm
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem removeOp_id (F : C ⥤ D) : NatTrans.removeOp (𝟙 F.op) = 𝟙 F :=
  rfl


/-- Given a natural transformation `α : F.unop ⟶ G.unop`, we can take the opposite of each
component obtaining a natural transformation `G ⟶ F`. -/
@[simps]
protected def removeUnop {F G : Cᵒᵖ ⥤ Dᵒᵖ} (α : F.unop ⟶ G.unop) : G ⟶ F where
  app X := (α.app (unop X)).op
  naturality X Y f :=
                              /-
                                C : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                D : Type u₂
                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                F✝ G✝ : CategoryTheory.Functor C D
                                F G : CategoryTheory.Functor (Opposite C) (Opposite D)
                                α : Quiver.Hom F.unop G.unop
                                X Y : Opposite C
                                f : Quiver.Hom X Y
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (α.app (Opposite …
                              -/
    Quiver.Hom.unop_inj <| by simpa only [Functor.unop_map] using (α.naturality f.unop).symm
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem removeUnop_id (F : Cᵒᵖ ⥤ Dᵒᵖ) : NatTrans.removeUnop (𝟙 F.unop) = 𝟙 F :=
  rfl


/-- Given a natural transformation `α : F ⟶ G`, for `F G : C ⥤ Dᵒᵖ`,
taking `unop` of each component gives a natural transformation `G.leftOp ⟶ F.leftOp`.
-/
@[simps]
protected def leftOp (α : F ⟶ G) : G.leftOp ⟶ F.leftOp where
  app X := (α.app (unop X)).unop
                                            /-
                                              C : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                              D : Type u₂
                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                              F G H : CategoryTheory.Functor C (Opposite D)
                                              α : Quiver.Hom F G
                                              X Y : Opposite C
                                              f : Quiver.Hom X Y
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.leftOp.map f) ((fun X => (α.app (O …
                                            -/
  naturality X Y f := Quiver.Hom.op_inj (by simp)
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem leftOp_id : NatTrans.leftOp (𝟙 F : F ⟶ F) = 𝟙 F.leftOp :=
  rfl


@[simp]
theorem leftOp_comp (α : F ⟶ G) (β : G ⟶ H) : NatTrans.leftOp (α ≫ β) =
    NatTrans.leftOp β ≫ NatTrans.leftOp α :=
  rfl


/-- Given a natural transformation `α : F.leftOp ⟶ G.leftOp`, for `F G : C ⥤ Dᵒᵖ`,
taking `op` of each component gives a natural transformation `G ⟶ F`.
-/
@[simps]
protected def removeLeftOp (α : F.leftOp ⟶ G.leftOp) : G ⟶ F where
  app X := (α.app (op X)).op
  naturality X Y f :=
                              /-
                                C : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                D : Type u₂
                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                F G H : CategoryTheory.Functor C (Opposite D)
                                α : Quiver.Hom F.leftOp G.leftOp
                                X Y : C
                                f : Quiver.Hom X Y
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (α.app { unop := …
                              -/
    Quiver.Hom.unop_inj <| by simpa only [Functor.leftOp_map] using (α.naturality f.op).symm
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem removeLeftOp_id : NatTrans.removeLeftOp (𝟙 F.leftOp) = 𝟙 F :=
  rfl


/-- Given a natural transformation `α : F ⟶ G`, for `F G : Cᵒᵖ ⥤ D`,
taking `op` of each component gives a natural transformation `G.rightOp ⟶ F.rightOp`.
-/
@[simps]
protected def rightOp (α : F ⟶ G) : G.rightOp ⟶ F.rightOp where
  app _ := (α.app _).op
                                              /-
                                                C : Type u₁
                                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                D : Type u₂
                                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                F G H : CategoryTheory.Functor (Opposite C) D
                                                α : Quiver.Hom F G
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.rightOp.map f) ((fun x => (α.app { …
                                              -/
  naturality X Y f := Quiver.Hom.unop_inj (by simp)
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem rightOp_id : NatTrans.rightOp (𝟙 F : F ⟶ F) = 𝟙 F.rightOp :=
  rfl


@[simp]
theorem rightOp_comp (α : F ⟶ G) (β : G ⟶ H) : NatTrans.rightOp (α ≫ β) =
    NatTrans.rightOp β ≫ NatTrans.rightOp α :=
  rfl


/-- Given a natural transformation `α : F.rightOp ⟶ G.rightOp`, for `F G : Cᵒᵖ ⥤ D`,
taking `unop` of each component gives a natural transformation `G ⟶ F`.
-/
@[simps]
protected def removeRightOp (α : F.rightOp ⟶ G.rightOp) : G ⟶ F where
  app X := (α.app X.unop).unop
  naturality X Y f :=
                            /-
                              C : Type u₁
                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                              D : Type u₂
                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                              F G H : CategoryTheory.Functor (Opposite C) D
                              α : Quiver.Hom F.rightOp G.rightOp
                              X Y : Opposite C
                              f : Quiver.Hom X Y
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (α.app (Opposite …
                            -/
    Quiver.Hom.op_inj <| by simpa only [Functor.rightOp_map] using (α.naturality f.unop).symm
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem removeRightOp_id : NatTrans.removeRightOp (𝟙 F.rightOp) = 𝟙 F :=
  rfl


/-- The opposite isomorphism.
-/
@[simps]
protected def op (α : X ≅ Y) : op Y ≅ op X where
  hom := α.hom.op
  inv := α.inv.op
  hom_inv_id := Quiver.Hom.unop_inj α.inv_hom_id
  inv_hom_id := Quiver.Hom.unop_inj α.hom_inv_id


/-- The isomorphism obtained from an isomorphism in the opposite category. -/
@[simps]
def unop {X Y : Cᵒᵖ} (f : X ≅ Y) : Y.unop ≅ X.unop where
  hom := f.hom.unop
  inv := f.inv.unop
                   /-
                     C : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                     X✝ Y✝ : C
                     X Y : Opposite C
                     f : CategoryTheory.Iso X Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.unop f.inv.unop) (CategoryTheor …
                   -/
  hom_inv_id := by simp only [← unop_comp, f.inv_hom_id, unop_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                     X✝ Y✝ : C
                     X Y : Opposite C
                     f : CategoryTheory.Iso X Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f.inv.unop f.hom.unop) (CategoryTheor …
                   -/
  inv_hom_id := by simp only [← unop_comp, f.hom_inv_id, unop_id]
                   /-
                     🎉 no goals
                   -/


@[simp]
                                                               /-
                                                                 C : Type u₁
                                                                 inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                 X Y : Opposite C
                                                                 f : CategoryTheory.Iso X Y
                                                                 ⊢ Eq f.unop.op f
                                                               -/
theorem unop_op {X Y : Cᵒᵖ} (f : X ≅ Y) : f.unop.op = f := by (ext; rfl)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
                                                             /-
                                                               C : Type u₁
                                                               inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                               X Y : C
                                                               f : CategoryTheory.Iso X Y
                                                               ⊢ Eq f.op.unop f
                                                             -/
theorem op_unop {X Y : C} (f : X ≅ Y) : f.op.unop = f := by (ext; rfl)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[reassoc (attr := simp)]
lemma unop_hom_inv_id_app : (e.hom.app X).unop ≫ (e.inv.app X).unop = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    F G : CategoryTheory.Functor C (Opposite D)
    e : CategoryTheory.Iso F G
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.hom.app X).unop (e.inv.app X).unop …
  -/
  rw [← unop_comp, inv_hom_id_app, unop_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma unop_inv_hom_id_app : (e.inv.app X).unop ≫ (e.hom.app X).unop = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} D
    F G : CategoryTheory.Functor C (Opposite D)
    e : CategoryTheory.Iso F G
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X).unop (e.hom.app X).unop …
  -/
  rw [← unop_comp, hom_inv_id_app, unop_id]
  /-
    🎉 no goals
  -/


/-- The natural isomorphism between opposite functors `G.op ≅ F.op` induced by a natural
isomorphism between the original functors `F ≅ G`. -/
@[simps]
protected def op (α : F ≅ G) : G.op ≅ F.op where
  hom := NatTrans.op α.hom
  inv := NatTrans.op α.inv
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F G : CategoryTheory.Functor C D
                     α : CategoryTheory.Iso F G
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatTrans.op α.hom) (C …
                   -/
  hom_inv_id := by ext; dsimp; rw [← op_comp]; rw [α.inv_hom_id_app]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F G : CategoryTheory.Functor C D
                     α : CategoryTheory.Iso F G
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatTrans.op α.inv) (C …
                   -/
  inv_hom_id := by ext; dsimp; rw [← op_comp]; rw [α.hom_inv_id_app]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The natural isomorphism between functors `G ≅ F` induced by a natural isomorphism
between the opposite functors `F.op ≅ G.op`. -/
@[simps]
protected def removeOp (α : F.op ≅ G.op) : G ≅ F where
  hom := NatTrans.removeOp α.hom
  inv := NatTrans.removeOp α.inv


/-- The natural isomorphism between functors `G.unop ≅ F.unop` induced by a natural isomorphism
between the original functors `F ≅ G`. -/
@[simps]
protected def unop {F G : Cᵒᵖ ⥤ Dᵒᵖ} (α : F ≅ G) : G.unop ≅ F.unop where
  hom := NatTrans.unop α.hom
  inv := NatTrans.unop α.inv


/-- An equivalence between categories gives an equivalence between the opposite categories.
-/
@[simps]
def op (e : C ≌ D) : Cᵒᵖ ≌ Dᵒᵖ where
  functor := e.functor.op
  inverse := e.inverse.op
  unitIso := (NatIso.op e.unitIso).symm
  counitIso := (NatIso.op e.counitIso).symm
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Equivalence C D
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.op.map ((CategoryTheory.Na …
    -/
    apply Quiver.Hom.unop_inj
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Equivalence C D
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.op.map ((CategoryTheory.Na …
    -/
    dsimp
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Equivalence C D
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.counitIso.inv.app (e.functor.obj ( …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- An equivalence between opposite categories gives an equivalence between the original categories.
-/
@[simps]
def unop (e : Cᵒᵖ ≌ Dᵒᵖ) : C ≌ D where
  functor := e.functor.unop
  inverse := e.inverse.unop
  unitIso := (NatIso.unop e.unitIso).symm
  counitIso := (NatIso.unop e.counitIso).symm
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Equivalence (Opposite C) (Opposite D)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.unop.map ((CategoryTheory. …
    -/
    apply Quiver.Hom.op_inj
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Equivalence (Opposite C) (Opposite D)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.unop.map ((CategoryTheory. …
    -/
    dsimp
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      e : CategoryTheory.Equivalence (Opposite C) (Opposite D)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.counitIso.inv.app (e.functor.obj { …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The equivalence between arrows of the form `A ⟶ B` and `B.unop ⟶ A.unop`. Useful for building
adjunctions.
Note that this (definitionally) gives variants
```
def opEquiv' (A : C) (B : Cᵒᵖ) : (Opposite.op A ⟶ B) ≃ (B.unop ⟶ A) :=
  opEquiv _ _

def opEquiv'' (A : Cᵒᵖ) (B : C) : (A ⟶ Opposite.op B) ≃ (B ⟶ A.unop) :=
  opEquiv _ _

def opEquiv''' (A B : C) : (Opposite.op A ⟶ Opposite.op B) ≃ (B ⟶ A) :=
  opEquiv _ _
```
-/
@[simps]
def opEquiv (A B : Cᵒᵖ) : (A ⟶ B) ≃ (B.unop ⟶ A.unop) where
  toFun f := f.unop
  invFun g := g.op
  left_inv _ := rfl
  right_inv _ := rfl


instance subsingleton_of_unop (A B : Cᵒᵖ) [Subsingleton (unop B ⟶ unop A)] : Subsingleton (A ⟶ B) :=
  (opEquiv A B).subsingleton


instance decidableEqOfUnop (A B : Cᵒᵖ) [DecidableEq (unop B ⟶ unop A)] : DecidableEq (A ⟶ B) :=
  (opEquiv A B).decidableEq


/-- The equivalence between isomorphisms of the form `A ≅ B` and `B.unop ≅ A.unop`.

Note this is definitionally the same as the other three variants:
* `(Opposite.op A ≅ B) ≃ (B.unop ≅ A)`
* `(A ≅ Opposite.op B) ≃ (B ≅ A.unop)`
* `(Opposite.op A ≅ Opposite.op B) ≃ (B ≅ A)`
-/
@[simps]
def isoOpEquiv (A B : Cᵒᵖ) : (A ≅ B) ≃ (B.unop ≅ A.unop) where
  toFun f := f.unop
  invFun g := g.op
  left_inv _ := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      A B : Opposite C
      x✝ : CategoryTheory.Iso A B
      ⊢ Eq ((fun g => g.op) ((fun f => f.unop) x✝)) x✝
    -/
    ext
    /-
      case w
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      A B : Opposite C
      x✝ : CategoryTheory.Iso A B
      ⊢ Eq ((fun g => g.op) ((fun f => f.unop) x✝)).hom x✝.hom
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv _ := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      A B : Opposite C
      x✝ : CategoryTheory.Iso (Opposite.unop B) (Opposite.unop A)
      ⊢ Eq ((fun f => f.unop) ((fun g => g.op) x✝)) x✝
    -/
    ext
    /-
      case w
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      A B : Opposite C
      x✝ : CategoryTheory.Iso (Opposite.unop B) (Opposite.unop A)
      ⊢ Eq ((fun f => f.unop) ((fun g => g.op) x✝)).hom x✝.hom
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The equivalence of functor categories induced by `op` and `unop`.
-/
@[simps]
def opUnopEquiv : (C ⥤ D)ᵒᵖ ≌ Cᵒᵖ ⥤ Dᵒᵖ where
  functor := opHom _ _
  inverse := opInv _ _
  unitIso :=
    NatIso.ofComponents (fun F => F.unop.opUnopIso.op)
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          ⊢ ∀ {X Y : Opposite (CategoryTheory.Functor C D)} (f : Quiver.Hom X Y), Eq (Ca …
        -/
        intro F G f
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor C D)
          f : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
        -/
        dsimp [opUnopIso]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor C D)
          f : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.NatIso.ofComponents …
        -/
        rw [show f = f.unop.op by simp, ← op_comp, ← op_comp]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor C D)
          f : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatIso.ofComponents ( …
        -/
        congr 1
        /-
          case e_f
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor C D)
          f : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatIso.ofComponents ( …
        -/
        aesop_cat)
        /-
          🎉 no goals
        -/
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                 ⊢ ∀ {X Y : CategoryTheory.Functor (Opposite C) (Opposite D)} (f : Quiver.Hom X …
               -/
  counitIso := NatIso.ofComponents fun F => F.unopOpIso
               /-
                 🎉 no goals
               -/


/-- The equivalence of functor categories induced by `leftOp` and `rightOp`.
-/
@[simps!]
def leftOpRightOpEquiv : (Cᵒᵖ ⥤ D)ᵒᵖ ≌ C ⥤ Dᵒᵖ where
  functor :=
    { obj := fun F => F.unop.rightOp
      map := fun η => NatTrans.rightOp η.unop }
  inverse :=
    { obj := fun F => op F.leftOp
      map := fun η => η.leftOp.op }
  unitIso :=
    NatIso.ofComponents (fun F => F.unop.rightOpLeftOpIso.op)
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          ⊢ ∀ {X Y : Opposite (CategoryTheory.Functor (Opposite C) D)} (f : Quiver.Hom X …
        -/
        intro F G η
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor (Opposite C) D)
          η : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor (Opposite C) D)
          η : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp η (Opposite.unop G).rightOpLeftOpIso. …
        -/
        rw [show η = η.unop.op by simp, ← op_comp, ← op_comp]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor (Opposite C) D)
          η : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop G).rightOpLeftOpIso.ho …
        -/
        congr 1
        /-
          case e_f
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F G : Opposite (CategoryTheory.Functor (Opposite C) D)
          η : Quiver.Hom F G
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop G).rightOpLeftOpIso.ho …
        -/
        aesop_cat)
        /-
          🎉 no goals
        -/
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                 ⊢ ∀ {X Y : CategoryTheory.Functor C (Opposite D)} (f : Quiver.Hom X Y), Eq (Ca …
               -/
  counitIso := NatIso.ofComponents fun F => F.leftOpRightOpIso
               /-
                 🎉 no goals
               -/


instance {F : C ⥤ D} [EssSurj F] : EssSurj F.op where
  mem_essImage X := ⟨op _, ⟨(F.objObjPreimageIso X.unop).op.symm⟩⟩


instance {F : Cᵒᵖ ⥤ D} [EssSurj F] : EssSurj F.rightOp where
  mem_essImage X := ⟨_, ⟨(F.objObjPreimageIso X.unop).op.symm⟩⟩


instance {F : C ⥤ Dᵒᵖ} [EssSurj F] : EssSurj F.leftOp where
  mem_essImage X := ⟨op _, ⟨(F.objObjPreimageIso (op X)).unop.symm⟩⟩


instance {F : C ⥤ D} [IsEquivalence F] : IsEquivalence F.op where


instance {F : Cᵒᵖ ⥤ D} [IsEquivalence F] : IsEquivalence F.rightOp where


instance {F : C ⥤ Dᵒᵖ} [IsEquivalence F] : IsEquivalence F.leftOp where


