/-- The bijection `ShiftedHom X Y n ≃ ShiftedHom (Opposite.op Y) (Opposite.op X) n` when
`n : ℤ`, and `X` and `Y` are objects of a category equipped with a shift by `ℤ`. -/
noncomputable def opEquiv (n : ℤ) :
    ShiftedHom X Y n ≃ ShiftedHom (Opposite.op Y) (Opposite.op X) n :=
  Quiver.Hom.opEquiv.trans
    ((opShiftFunctorEquivalence C n).symm.toAdjunction.homEquiv (Opposite.op Y) (Opposite.op X))


lemma opEquiv_symm_apply {n : ℤ} (f : ShiftedHom (Opposite.op Y) (Opposite.op X) n) :
    (opEquiv n).symm f =
      ((opShiftFunctorEquivalence C n).unitIso.inv.app (Opposite.op X)).unop ≫ f.unop⟦n⟧' := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y : C
    n : Int
    f : CategoryTheory.ShiftedHom { unop := Y } { unop := X } n
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv n).symm f) (CategoryTheory.CategorySt …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma opEquiv_symm_apply_comp {X Y : C} {a : ℤ}
    (f : ShiftedHom (Opposite.op X) (Opposite.op Y) a) {b : ℤ} {Z : C}
    (z : ShiftedHom X Z b) {c : ℤ} (h : b + a = c) :
    ((ShiftedHom.opEquiv a).symm f).comp z h =
      (ShiftedHom.opEquiv a).symm (z.op ≫ f) ≫
        (shiftFunctorAdd' C b a c h).inv.app Z := by
  rw [ShiftedHom.opEquiv_symm_apply, ShiftedHom.opEquiv_symm_apply,
    ShiftedHom.comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y : C
    a : Int
    f : CategoryTheory.ShiftedHom { unop := X } { unop := Y } a
    b : Int
    Z : C
    z : CategoryTheory.ShiftedHom X Z b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y : C
    a : Int
    f : CategoryTheory.ShiftedHom { unop := X } { unop := Y } a
    b : Int
    Z : C
    z : CategoryTheory.ShiftedHom X Z b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, Functor.map_comp]
  /-
    🎉 no goals
  -/


lemma opEquiv_symm_comp {a b : ℤ}
    (f : ShiftedHom (Opposite.op Z) (Opposite.op Y) a)
    (g : ShiftedHom (Opposite.op Y) (Opposite.op X) b)
    {c : ℤ} (h : b + a = c) :
    (opEquiv _).symm (f.comp g h) =
                                                         /-
                                                           C : Type u_1
                                                           inst✝¹ : CategoryTheory.Category.{?u.5890, u_1} C
                                                           inst✝ : CategoryTheory.HasShift C Int
                                                           X Y Z : C
                                                           a b : Int
                                                           f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
                                                           g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
                                                           c : Int
                                                           h : Eq (HAdd.hAdd b a) c
                                                           ⊢ Eq (HAdd.hAdd a b) c
                                                         -/
      ((opEquiv _).symm g).comp ((opEquiv _).symm f) (by omega) := by
                                                         /-
                                                           🎉 no goals
                                                         -/
  rw [opEquiv_symm_apply, opEquiv_symm_apply,
    opShiftFunctorEquivalence_unitIso_inv_app_eq _ _ _ _ (show a + b = c by omega), comp, comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    a b : Int
    f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
    g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  rw [assoc, assoc, assoc, assoc, ← Functor.map_comp, ← unop_comp_assoc,
    Iso.inv_hom_id_app]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    a b : Int
    f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
    g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  dsimp
  rw [assoc, id_comp, Functor.map_comp, ← NatTrans.naturality_assoc,
    ← NatTrans.naturality, opEquiv_symm_apply]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    a b : Int
    f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
    g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  dsimp
  rw [← Functor.map_comp_assoc, ← Functor.map_comp_assoc,
    ← Functor.map_comp_assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    a b : Int
    f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
    g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  rw [← unop_comp_assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    a b : Int
    f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
    g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  erw [← NatTrans.naturality]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    a b : Int
    f : CategoryTheory.ShiftedHom { unop := Z } { unop := Y } a
    g : CategoryTheory.ShiftedHom { unop := Y } { unop := X } b
    c : Int
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The bijection `ShiftedHom X Y a' ≃ (Opposite.op (Y⟦a⟧) ⟶ (Opposite.op X)⟦n⟧)`
when integers `n`, `a` and `a'` satisfy `n + a = a'`, and `X` and `Y` are objects
of a category equipped with a shift by `ℤ`. -/
noncomputable def opEquiv' (n a a' : ℤ) (h : n + a = a') :
    ShiftedHom X Y a' ≃ (Opposite.op (Y⟦a⟧) ⟶ (Opposite.op X)⟦n⟧) :=
                                  /-
                                    C : Type u_1
                                    inst✝¹ : CategoryTheory.Category.{?u.50597, u_1} C
                                    inst✝ : CategoryTheory.HasShift C Int
                                    X Y Z : C
                                    n a a' : Int
                                    h : Eq (HAdd.hAdd n a) a'
                                    ⊢ Eq (HAdd.hAdd a n) a'
                                  -/
  ((shiftFunctorAdd' C a n a' (by omega)).symm.app Y).homToEquiv.symm.trans (opEquiv n)
                                  /-
                                    🎉 no goals
                                  -/


lemma opEquiv'_symm_apply {n a : ℤ} (f : Opposite.op (Y⟦a⟧) ⟶ (Opposite.op X)⟦n⟧)
    (a' : ℤ) (h : n + a = a') :
    (opEquiv' n a a' h).symm f =
                                                          /-
                                                            C : Type u_1
                                                            inst✝¹ : CategoryTheory.Category.{?u.52631, u_1} C
                                                            inst✝ : CategoryTheory.HasShift C Int
                                                            X Y Z : C
                                                            n a : Int
                                                            f : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((CategoryT …
                                                            a' : Int
                                                            h : Eq (HAdd.hAdd n a) a'
                                                            ⊢ Eq (HAdd.hAdd a n) a'
                                                          -/
      (opEquiv n).symm f ≫ (shiftFunctorAdd' C a n a' (by omega)).inv.app _ :=
                                                          /-
                                                            🎉 no goals
                                                          -/
  rfl


lemma opEquiv'_apply {a' : ℤ} (f : ShiftedHom X Y a') (n a : ℤ) (h : n + a = a') :
    opEquiv' n a a' h f =
                                                    /-
                                                      C : Type u_1
                                                      inst✝¹ : CategoryTheory.Category.{?u.55307, u_1} C
                                                      inst✝ : CategoryTheory.HasShift C Int
                                                      X Y Z : C
                                                      a' : Int
                                                      f : CategoryTheory.ShiftedHom X Y a'
                                                      n a : Int
                                                      h : Eq (HAdd.hAdd n a) a'
                                                      ⊢ Eq (HAdd.hAdd a n) a'
                                                    -/
      opEquiv n (f ≫ (shiftFunctorAdd' C a n a' (by omega)).hom.app Y) := by
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y : C
    a' : Int
    f : CategoryTheory.ShiftedHom X Y a'
    n a : Int
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv' n a a' h) f) ((CategoryTheory.Shifte …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma opEquiv'_symm_op_opShiftFunctorEquivalence_counitIso_inv_app_op_shift
    {n m : ℤ} (f : ShiftedHom X Y n) (g : ShiftedHom Y Z m)
    (q : ℤ) (hq : n + m = q) :
    (opEquiv' n m q hq).symm
        (g.op ≫ (opShiftFunctorEquivalence C n).counitIso.inv.app _ ≫ f.op⟦n⟧') =
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.56606, u_1} C
                     inst✝ : CategoryTheory.HasShift C Int
                     X Y Z : C
                     n m : Int
                     f : CategoryTheory.ShiftedHom X Y n
                     g : CategoryTheory.ShiftedHom Y Z m
                     q : Int
                     hq : Eq (HAdd.hAdd n m) q
                     ⊢ Eq (HAdd.hAdd m n) q
                   -/
      f.comp g (by omega) := by
                   /-
                     🎉 no goals
                   -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    n m : Int
    f : CategoryTheory.ShiftedHom X Y n
    g : CategoryTheory.ShiftedHom Y Z m
    q : Int
    hq : Eq (HAdd.hAdd n m) q
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv' n m q hq).symm (CategoryTheory.Categ …
  -/
  rw [opEquiv'_symm_apply, opEquiv_symm_apply]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    n m : Int
    f : CategoryTheory.ShiftedHom X Y n
    g : CategoryTheory.ShiftedHom Y Z m
    q : Int
    hq : Eq (HAdd.hAdd n m) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    n m : Int
    f : CategoryTheory.ShiftedHom X Y n
    g : CategoryTheory.ShiftedHom Y Z m
    q : Int
    hq : Eq (HAdd.hAdd n m) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply Quiver.Hom.op_inj
  simp only [assoc, Functor.map_comp, op_comp, Quiver.Hom.op_unop,
    opShiftFunctorEquivalence_unitIso_inv_naturality]
  /-
    case a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y Z : C
    n m : Int
    f : CategoryTheory.ShiftedHom X Y n
    g : CategoryTheory.ShiftedHom Y Z m
    q : Int
    hq : Eq (HAdd.hAdd n m) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd' C m …
  -/
  erw [(opShiftFunctorEquivalence C n).inverse_counitInv_comp_assoc (Opposite.op Y)]
  /-
    🎉 no goals
  -/


lemma opEquiv'_symm_comp (f : Y ⟶ X) {n a : ℤ} (x : Opposite.op (Z⟦a⟧) ⟶ (Opposite.op X⟦n⟧))
    (a' : ℤ) (h : n + a = a') :
    (opEquiv' n a a' h).symm (x ≫ f.op⟦n⟧') = f ≫ (opEquiv' n a a' h).symm x :=
                        /-
                          C : Type u_1
                          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                          inst✝ : CategoryTheory.HasShift C Int
                          X Y Z : C
                          f : Quiver.Hom Y X
                          n a : Int
                          x : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Z } ((CategoryT …
                          a' : Int
                          h : Eq (HAdd.hAdd n a) a'
                          ⊢ Eq (Quiver.Hom.op ((CategoryTheory.ShiftedHom.opEquiv' n a a' h).symm (Categ …
                        -/
  Quiver.Hom.op_inj (by simp [opEquiv'_symm_apply, opEquiv_symm_apply])
                        /-
                          🎉 no goals
                        -/


lemma opEquiv'_zero_add_symm (a : ℤ) (f : Opposite.op (Y⟦a⟧) ⟶ (Opposite.op X)⟦(0 : ℤ)⟧) :
    (opEquiv' 0 a a (zero_add a)).symm f =
      ((shiftFunctorZero Cᵒᵖ ℤ).hom.app _).unop ≫ f.unop := by
  simp [opEquiv'_symm_apply, opEquiv_symm_apply, shiftFunctorAdd'_add_zero,
    opShiftFunctorEquivalence_zero_unitIso_inv_app]


lemma opEquiv'_add_symm (n m a a' a'' : ℤ) (ha' : n + a = a') (ha'' : m + a' = a'')
    (x : (Opposite.op (Y⟦a⟧) ⟶ (Opposite.op X)⟦m + n⟧)) :
                                /-
                                  C : Type u_1
                                  inst✝¹ : CategoryTheory.Category.{?u.76239, u_1} C
                                  inst✝ : CategoryTheory.HasShift C Int
                                  X Y Z : C
                                  n m a a' a'' : Int
                                  ha' : Eq (HAdd.hAdd n a) a'
                                  ha'' : Eq (HAdd.hAdd m a') a''
                                  x : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((CategoryT …
                                  ⊢ Eq (HAdd.hAdd (HAdd.hAdd m n) a) a''
                                -/
    (opEquiv' (m + n) a a'' (by omega)).symm x =
                                /-
                                  🎉 no goals
                                -/
      (opEquiv' m a' a'' ha'').symm ((opEquiv' n a a' ha').symm
        (x ≫ (shiftFunctorAdd Cᵒᵖ m n).hom.app _)).op := by
  simp only [opEquiv'_symm_apply, opEquiv_symm_apply,
    opShiftFunctorEquivalence_unitIso_inv_app_eq _ _ _ _ (add_comm n m)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y : C
    n m a a' a'' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    x : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((CategoryT …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  simp only [assoc, Functor.map_comp, ← shiftFunctorAdd'_eq_shiftFunctorAdd,
    ← NatTrans.naturality_assoc,
    shiftFunctorAdd'_assoc_inv_app a n m a' (m + n) a'' (by omega) (by omega) (by omega)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X Y : C
    n m a a' a'' : Int
    ha' : Eq (HAdd.hAdd n a) a'
    ha'' : Eq (HAdd.hAdd m a') a''
    x : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((CategoryT …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma opEquiv_symm_add {n : ℤ} (x y : ShiftedHom (Opposite.op Y) (Opposite.op X) n) :
    (opEquiv n).symm (x + y) = (opEquiv n).symm x + (opEquiv n).symm y := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n : Int
    x y : CategoryTheory.ShiftedHom { unop := Y } { unop := X } n
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv n).symm (HAdd.hAdd x y)) (HAdd.hAdd ( …
  -/
  dsimp [opEquiv_symm_apply]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n : Int
    x y : CategoryTheory.ShiftedHom { unop := Y } { unop := X } n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  rw [← Preadditive.comp_add, ← Functor.map_add]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n : Int
    x y : CategoryTheory.ShiftedHom { unop := Y } { unop := X } n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma opEquiv'_symm_add {n a : ℤ} (x y : (Opposite.op (Y⟦a⟧) ⟶ (Opposite.op X)⟦n⟧))
    (a' : ℤ) (h : n + a = a') :
    (opEquiv' n a a' h).symm (x + y) =
      (opEquiv' n a a' h).symm x + (opEquiv' n a a' h).symm y := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n a : Int
    x y : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((Categor …
    a' : Int
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv' n a a' h).symm (HAdd.hAdd x y)) (HAd …
  -/
  dsimp [opEquiv']
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n a : Int
    x y : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((Categor …
    a' : Int
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' C a n a' ⋯).symm.app Y).homToEquiv ((C …
  -/
  erw [opEquiv_symm_add, Iso.homToEquiv_apply, Iso.homToEquiv_apply, Iso.homToEquiv_apply]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n a : Int
    x y : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((Categor …
    a' : Int
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd ((CategoryTheory.ShiftedHo …
  -/
  rw [Preadditive.add_comp]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    X Y : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    n a : Int
    x y : Quiver.Hom { unop := (CategoryTheory.shiftFunctor C a).obj Y } ((Categor …
    a' : Int
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ShiftedHo …
  -/
  rfl
  /-
    🎉 no goals
  -/


