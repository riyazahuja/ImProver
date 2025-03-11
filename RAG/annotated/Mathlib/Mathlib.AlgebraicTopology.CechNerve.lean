/-- The Čech nerve associated to an arrow. -/
@[simps]
def cechNerve : SimplicialObject C where
  obj n := widePullback.{0} f.right (fun _ : Fin (n.unop.len + 1) => f.left) fun _ => f.hom
  map g := WidePullback.lift (WidePullback.base _)
                                                          /-
                                                            C : Type u
                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                            f : CategoryTheory.Arrow C
                                                            inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
                                                            X✝ Y✝ : Opposite SimplexCategory
                                                            g : Quiver.Hom X✝ Y✝
                                                            ⊢ ∀ (j : Fin (HAdd.hAdd (Opposite.unop Y✝).len 1)), Eq (CategoryTheory.Categor …
                                                          -/
    (fun i => WidePullback.π _ (g.unop.toOrderHom i)) (by aesop_cat)
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- The morphism between Čech nerves associated to a morphism of arrows. -/
@[simps]
def mapCechNerve {f g : Arrow C}
    [∀ n : ℕ, HasWidePullback f.right (fun _ : Fin (n + 1) => f.left) fun _ => f.hom]
    [∀ n : ℕ, HasWidePullback g.right (fun _ : Fin (n + 1) => g.left) fun _ => g.hom] (F : f ⟶ g) :
    f.cechNerve ⟶ g.cechNerve where
  app n :=
    WidePullback.lift (WidePullback.base _ ≫ F.right) (fun i => WidePullback.π _ i ≫ F.left)
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    f✝ : CategoryTheory.Arrow C
                    inst✝² : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f✝.right (fun x => …
                    f g : CategoryTheory.Arrow C
                    inst✝¹ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x =>  …
                    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback g.right (fun x => g …
                    F : Quiver.Hom f g
                    n : Opposite SimplexCategory
                    j : Fin (HAdd.hAdd (Opposite.unop n).len 1)
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => CategoryTheory.CategoryStr …
                  -/
      fun j => by simp
                  /-
                    🎉 no goals
                  -/


/-- The augmented Čech nerve associated to an arrow. -/
@[simps]
def augmentedCechNerve : SimplicialObject.Augmented C where
  left := f.cechNerve
  right := f.right
  hom := { app := fun _ => WidePullback.base _ }


/-- The morphism between augmented Čech nerve associated to a morphism of arrows. -/
@[simps]
def mapAugmentedCechNerve {f g : Arrow C}
    [∀ n : ℕ, HasWidePullback f.right (fun _ : Fin (n + 1) => f.left) fun _ => f.hom]
    [∀ n : ℕ, HasWidePullback g.right (fun _ : Fin (n + 1) => g.left) fun _ => g.hom] (F : f ⟶ g) :
    f.augmentedCechNerve ⟶ g.augmentedCechNerve where
  left := mapCechNerve F
  right := F.right


/-- The Čech nerve construction, as a functor from `Arrow C`. -/
@[simps]
def cechNerve : Arrow C ⥤ SimplicialObject C where
  obj f := f.cechNerve
  map F := Arrow.mapCechNerve F


/-- The augmented Čech nerve construction, as a functor from `Arrow C`. -/
@[simps!]
def augmentedCechNerve : Arrow C ⥤ SimplicialObject.Augmented C where
  obj f := f.augmentedCechNerve
  map F := Arrow.mapAugmentedCechNerve F


/-- A helper function used in defining the Čech adjunction. -/
@[simps]
def equivalenceRightToLeft (X : SimplicialObject.Augmented C) (F : Arrow C)
    (G : X ⟶ F.augmentedCechNerve) : Augmented.toArrow.obj X ⟶ F where
  left := G.left.app _ ≫ WidePullback.π _ 0
  right := G.right
  w := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      G : Quiver.Hom X F.augmentedCechNerve
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map (C …
    -/
    have := G.w
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      G : Quiver.Hom X F.augmentedCechNerve
      this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Cat …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map (C …
    -/
    apply_fun fun e => e.app (Opposite.op <| SimplexCategory.mk 0) at this
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      G : Quiver.Hom X F.augmentedCechNerve
      this : Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Ca …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map (C …
    -/
    simpa using this
    /-
      🎉 no goals
    -/


/-- A helper function used in defining the Čech adjunction. -/
@[simps]
def equivalenceLeftToRight (X : SimplicialObject.Augmented C) (F : Arrow C)
    (G : Augmented.toArrow.obj X ⟶ F) : X ⟶ F.augmentedCechNerve where
  left :=
    { app := fun x =>
        Limits.WidePullback.lift (X.hom.app _ ≫ G.right)
          (fun i => X.left.map (SimplexCategory.const _ x.unop i).op ≫ G.left) fun i => by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            X : CategoryTheory.SimplicialObject.Augmented C
            F : CategoryTheory.Arrow C
            G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
            x : Opposite SimplexCategory
            i : Fin (HAdd.hAdd (Opposite.unop x).len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => CategoryTheory.CategoryStr …
          -/
          dsimp
          erw [Category.assoc, Arrow.w, Augmented.toArrow_obj_hom, NatTrans.naturality_assoc,
            Functor.const_obj_map, Category.id_comp]
      naturality := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
          ⊢ ∀ ⦃X_1 Y : Opposite SimplexCategory⦄ (f : Quiver.Hom X_1 Y), Eq (CategoryThe …
        -/
        intro x y f
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
          x y : Opposite SimplexCategory
          f : Quiver.Hom x y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.left.map f) ((fun x => CategoryThe …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
          x y : Opposite SimplexCategory
          f : Quiver.Hom x y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.left.map f) (CategoryTheory.Limits …
        -/
        ext
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            X : CategoryTheory.SimplicialObject.Augmented C
            F : CategoryTheory.Arrow C
            G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
            x y : Opposite SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd (Opposite.unop y).len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · dsimp
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            X : CategoryTheory.SimplicialObject.Augmented C
            F : CategoryTheory.Arrow C
            G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
            x y : Opposite SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd (Opposite.unop y).len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp only [WidePullback.lift_π, Category.assoc, ← X.left.map_comp_assoc]
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            X : CategoryTheory.SimplicialObject.Augmented C
            F : CategoryTheory.Arrow C
            G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
            x y : Opposite SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd (Opposite.unop y).len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.left.map (CategoryTheory.CategoryS …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            X : CategoryTheory.SimplicialObject.Augmented C
            F : CategoryTheory.Arrow C
            G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
            x y : Opposite SimplexCategory
            f : Quiver.Hom x y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · dsimp
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            X : CategoryTheory.SimplicialObject.Augmented C
            F : CategoryTheory.Arrow C
            G : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
            x y : Opposite SimplexCategory
            f : Quiver.Hom x y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp }
          /-
            🎉 no goals
          -/
  right := G.right


/-- A helper function used in defining the Čech adjunction. -/
@[simps]
def cechNerveEquiv (X : SimplicialObject.Augmented C) (F : Arrow C) :
    (Augmented.toArrow.obj X ⟶ F) ≃ (X ⟶ F.augmentedCechNerve) where
  toFun := equivalenceLeftToRight _ _
  invFun := equivalenceRightToLeft _ _
  left_inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      ⊢ Function.LeftInverse (CategoryTheory.SimplicialObject.equivalenceRightToLeft …
    -/
    intro A
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
      ⊢ Eq (CategoryTheory.SimplicialObject.equivalenceRightToLeft X F (CategoryTheo …
    -/
    ext
      /-
        case h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq (CategoryTheory.SimplicialObject.equivalenceRightToLeft X F (CategoryTheo …
      -/
    · dsimp
      /-
        case h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
      -/
      rw [WidePullback.lift_π]
      /-
        case h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.left.map ((SimplexCategory.mk 0).c …
      -/
      nth_rw 2 [← Category.id_comp A.left]
      /-
        case h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.left.map ((SimplexCategory.mk 0).c …
      -/
      congr 1
      /-
        case h₁.e_a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq (X.left.map ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0).op) ( …
      -/
      convert X.left.map_id _
      /-
        case h.e'_2.h.h.e'_8
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0).op (CategoryTheor …
      -/
      rw [← op_id]
      /-
        case h.e'_2.h.h.e'_8
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0).op (CategoryTheor …
      -/
      congr 1
      /-
        case h.e'_2.h.h.e'_8.e_f
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0) (CategoryTheory.C …
      -/
      ext ⟨a, ha⟩
      /-
        case h.e'_2.h.h.e'_8.e_f.a.h.h.mk.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        a : Nat
        ha : LT.lt a (HAdd.hAdd (SimplexCategory.mk 0).len 1)
        ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom ((SimplexCategory.mk 0).const (SimplexC …
      -/
      change a < 1 at ha
      /-
        case h.e'_2.h.h.e'_8.e_f.a.h.h.mk.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        a : Nat
        ha : LT.lt a 1
        ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom ((SimplexCategory.mk 0).const (SimplexC …
      -/
      change 0 = a
      /-
        case h.e'_2.h.h.e'_8.e_f.a.h.h.mk.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        a : Nat
        ha : LT.lt a 1
        ⊢ Eq 0 a
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) F
        ⊢ Eq (CategoryTheory.SimplicialObject.equivalenceRightToLeft X F (CategoryTheo …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      ⊢ Function.RightInverse (CategoryTheory.SimplicialObject.equivalenceRightToLef …
    -/
    intro A
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      X : CategoryTheory.SimplicialObject.Augmented C
      F : CategoryTheory.Arrow C
      A : Quiver.Hom X F.augmentedCechNerve
      ⊢ Eq (CategoryTheory.SimplicialObject.equivalenceLeftToRight X F (CategoryTheo …
    -/
    ext x : 2
      /-
        case h₁.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom X F.augmentedCechNerve
        x : Opposite SimplexCategory
        ⊢ Eq ((CategoryTheory.SimplicialObject.equivalenceLeftToRight X F (CategoryThe …
      -/
    · refine WidePullback.hom_ext _ _ _ (fun j => ?_) ?_
        /-
          case h₁.h.refine_1
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          A : Quiver.Hom X F.augmentedCechNerve
          x : Opposite SimplexCategory
          j : Fin (HAdd.hAdd (Opposite.unop x).len 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialObject.equ …
        -/
      · dsimp
        /-
          case h₁.h.refine_1
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          A : Quiver.Hom X F.augmentedCechNerve
          x : Opposite SimplexCategory
          j : Fin (HAdd.hAdd (Opposite.unop x).len 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
        -/
        simp
        /-
          case h₁.h.refine_1
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          A : Quiver.Hom X F.augmentedCechNerve
          x : Opposite SimplexCategory
          j : Fin (HAdd.hAdd (Opposite.unop x).len 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.left.app x) (CategoryTheory.Limits …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case h₁.h.refine_2
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          F : CategoryTheory.Arrow C
          A : Quiver.Hom X F.augmentedCechNerve
          x : Opposite SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialObject.equ …
        -/
      · simpa using congr_app A.w.symm x
        /-
          🎉 no goals
        -/
      /-
        case h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        X : CategoryTheory.SimplicialObject.Augmented C
        F : CategoryTheory.Arrow C
        A : Quiver.Hom X F.augmentedCechNerve
        ⊢ Eq (CategoryTheory.SimplicialObject.equivalenceLeftToRight X F (CategoryTheo …
      -/
    · rfl
      /-
        🎉 no goals
      -/


/-- The augmented Čech nerve construction is right adjoint to the `toArrow` functor. -/
abbrev cechNerveAdjunction : (Augmented.toArrow : _ ⥤ Arrow C) ⊣ augmentedCechNerve :=
  Adjunction.mkOfHomEquiv
    { homEquiv := cechNerveEquiv
                                          /-
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
                                            ⊢ ∀ {X' X : CategoryTheory.SimplicialObject.Augmented C} {Y : CategoryTheory.A …
                                          -/
      homEquiv_naturality_left_symm := by dsimp [cechNerveEquiv]; aesop_cat
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
      homEquiv_naturality_right := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          ⊢ ∀ {X : CategoryTheory.SimplicialObject.Augmented C} {Y Y' : CategoryTheory.A …
        -/
        dsimp [cechNerveEquiv]
        -- The next three lines were not needed before https://github.com/leanprover/lean4/pull/2644
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          ⊢ ∀ {X : CategoryTheory.SimplicialObject.Augmented C} {Y Y' : CategoryTheory.A …
        -/
        intro X Y Y' f g
        change equivalenceLeftToRight X Y' (f ≫ g) =
          equivalenceLeftToRight X Y f ≫ augmentedCechNerve.map g
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          X : CategoryTheory.SimplicialObject.Augmented C
          Y Y' : CategoryTheory.Arrow C
          f : Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.toArrow.obj X) Y
          g : Quiver.Hom Y Y'
          ⊢ Eq (CategoryTheory.SimplicialObject.equivalenceLeftToRight X Y' (CategoryThe …
        -/
        aesop_cat
        /-
          🎉 no goals
        -/
    }


/-- The Čech conerve associated to an arrow. -/
@[simps]
def cechConerve : CosimplicialObject C where
  obj n := widePushout f.left (fun _ : Fin (n.len + 1) => f.right) fun _ => f.hom
  map {x y} g := by
    refine WidePushout.desc (WidePushout.head _)
      (fun i => (@WidePushout.ι _ _ _ _ _ (fun _ => f.hom) (_) (g.toOrderHom i))) (fun j => ?_)
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout f.left (fun x => f.r …
      x y : SimplexCategory
      g : Quiver.Hom x y
      j : Fin (HAdd.hAdd x.len 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom ((fun i => CategoryTheory.Limit …
    -/
    rw [← WidePushout.arrow_ι]
    /-
      🎉 no goals
    -/


/-- The morphism between Čech conerves associated to a morphism of arrows. -/
@[simps]
def mapCechConerve {f g : Arrow C}
    [∀ n : ℕ, HasWidePushout f.left (fun _ : Fin (n + 1) => f.right) fun _ => f.hom]
    [∀ n : ℕ, HasWidePushout g.left (fun _ : Fin (n + 1) => g.right) fun _ => g.hom] (F : f ⟶ g) :
    f.cechConerve ⟶ g.cechConerve where
  app n := WidePushout.desc (F.left ≫ WidePushout.head _)
                            /-
                              C : Type u
                              inst✝³ : CategoryTheory.Category.{v, u} C
                              f✝ : CategoryTheory.Arrow C
                              inst✝² : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout f✝.left (fun x => f …
                              f g : CategoryTheory.Arrow C
                              inst✝¹ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout f.left (fun x => f. …
                              inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout g.left (fun x => g.r …
                              F : Quiver.Hom f g
                              n : SimplexCategory
                              i : Fin (HAdd.hAdd n.len 1)
                              ⊢ Quiver.Hom g.right (g.cechConerve.obj n)
                            -/
    (fun i => F.right ≫ (by apply WidePushout.ι _ i))
                            /-
                              🎉 no goals
                            -/
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    f✝ : CategoryTheory.Arrow C
                    inst✝² : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout f✝.left (fun x => f …
                    f g : CategoryTheory.Arrow C
                    inst✝¹ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout f.left (fun x => f. …
                    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePushout g.left (fun x => g.r …
                    F : Quiver.Hom f g
                    n : SimplexCategory
                    i : Fin (HAdd.hAdd n.len 1)
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom ((fun i => CategoryTheory.Categ …
                  -/
    (fun i => (by rw [← Arrow.w_assoc F, ← WidePushout.arrow_ι]))
                  /-
                    🎉 no goals
                  -/


/-- The augmented Čech conerve associated to an arrow. -/
@[simps]
def augmentedCechConerve : CosimplicialObject.Augmented C where
  left := f.left
  right := f.cechConerve
  hom :=
    { app := fun _ => (WidePushout.head _ : f.left ⟶ _) }


/-- The morphism between augmented Čech conerves associated to a morphism of arrows. -/
@[simps]
def mapAugmentedCechConerve {f g : Arrow C}
    [∀ n : ℕ, HasWidePushout f.left (fun _ : Fin (n + 1) => f.right) fun _ => f.hom]
    [∀ n : ℕ, HasWidePushout g.left (fun _ : Fin (n + 1) => g.right) fun _ => g.hom] (F : f ⟶ g) :
    f.augmentedCechConerve ⟶ g.augmentedCechConerve where
  left := F.left
  right := mapCechConerve F


/-- The Čech conerve construction, as a functor from `Arrow C`. -/
@[simps]
def cechConerve : Arrow C ⥤ CosimplicialObject C where
  obj f := f.cechConerve
  map F := Arrow.mapCechConerve F


/-- The augmented Čech conerve construction, as a functor from `Arrow C`. -/
@[simps]
def augmentedCechConerve : Arrow C ⥤ CosimplicialObject.Augmented C where
  obj f := f.augmentedCechConerve
  map F := Arrow.mapAugmentedCechConerve F


/-- A helper function used in defining the Čech conerve adjunction. -/
@[simps]
def equivalenceLeftToRight (F : Arrow C) (X : CosimplicialObject.Augmented C)
    (G : F.augmentedCechConerve ⟶ X) : F ⟶ Augmented.toArrow.obj X where
  left := G.left
  right := (WidePushout.ι _ 0 ≫ G.right.app (SimplexCategory.mk 0) : _)
  w := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      F : CategoryTheory.Arrow C
      X : CategoryTheory.CosimplicialObject.Augmented C
      G : Quiver.Hom F.augmentedCechConerve X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map G. …
    -/
    dsimp
    rw [@WidePushout.arrow_ι_assoc _ _ _ _ _ (fun (_ : Fin 1) => F.hom)
      (by dsimp; infer_instance)]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      F : CategoryTheory.Arrow C
      X : CategoryTheory.CosimplicialObject.Augmented C
      G : Quiver.Hom F.augmentedCechConerve X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp G.left (X.hom.app (SimplexCategory.mk …
    -/
    exact congr_app G.w (SimplexCategory.mk 0)
    /-
      🎉 no goals
    -/


/-- A helper function used in defining the Čech conerve adjunction. -/
@[simps!]
def equivalenceRightToLeft (F : Arrow C) (X : CosimplicialObject.Augmented C)
    (G : F ⟶ Augmented.toArrow.obj X) : F.augmentedCechConerve ⟶ X where
  left := G.left
  right :=
    { app := fun x =>
        Limits.WidePushout.desc (G.left ≫ X.hom.app _)
          (fun i => G.right ≫ X.right.map (SimplexCategory.const _ x i))
          (by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
              F : CategoryTheory.Arrow C
              X : CategoryTheory.CosimplicialObject.Augmented C
              G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
              x : SimplexCategory
              ⊢ ∀ (j : Fin (HAdd.hAdd x.len 1)), Eq (CategoryTheory.CategoryStruct.comp F.ho …
            -/
            rintro j
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
              F : CategoryTheory.Arrow C
              X : CategoryTheory.CosimplicialObject.Augmented C
              G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
              x : SimplexCategory
              j : Fin (HAdd.hAdd x.len 1)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp F.hom ((fun i => CategoryTheory.Categ …
            -/
            rw [← Arrow.w_assoc G]
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
              F : CategoryTheory.Arrow C
              X : CategoryTheory.CosimplicialObject.Augmented C
              G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
              x : SimplexCategory
              j : Fin (HAdd.hAdd x.len 1)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp G.left (CategoryTheory.CategoryStruct …
            -/
            have t := X.hom.naturality (SimplexCategory.const (SimplexCategory.mk 0) x j)
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
              F : CategoryTheory.Arrow C
              X : CategoryTheory.CosimplicialObject.Augmented C
              G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
              x : SimplexCategory
              j : Fin (HAdd.hAdd x.len 1)
              t : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CosimplicialObjec …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp G.left (CategoryTheory.CategoryStruct …
            -/
            dsimp at t ⊢
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
              F : CategoryTheory.Arrow C
              X : CategoryTheory.CosimplicialObject.Augmented C
              G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
              x : SimplexCategory
              j : Fin (HAdd.hAdd x.len 1)
              t : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp G.left (CategoryTheory.CategoryStruct …
            -/
            simp only [Category.id_comp] at t
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
              F : CategoryTheory.Arrow C
              X : CategoryTheory.CosimplicialObject.Augmented C
              G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
              x : SimplexCategory
              j : Fin (HAdd.hAdd x.len 1)
              t : Eq (X.hom.app x) (CategoryTheory.CategoryStruct.comp (X.hom.app (SimplexCa …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp G.left (CategoryTheory.CategoryStruct …
            -/
            rw [← t])
            /-
              🎉 no goals
            -/
      naturality := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
          ⊢ ∀ ⦃X_1 Y : SimplexCategory⦄ (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.Categ …
        -/
        intro x y f
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
          x y : SimplexCategory
          f : Quiver.Hom x y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.augmentedCechConerve.right.map f)  …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
          x y : SimplexCategory
          f : Quiver.Hom x y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.de …
        -/
        ext
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            F : CategoryTheory.Arrow C
            X : CategoryTheory.CosimplicialObject.Augmented C
            G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
            x y : SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd x.len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.ι  …
          -/
        · dsimp
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            F : CategoryTheory.Arrow C
            X : CategoryTheory.CosimplicialObject.Augmented C
            G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
            x y : SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd x.len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.ι  …
          -/
          simp only [WidePushout.ι_desc_assoc, WidePushout.ι_desc]
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            F : CategoryTheory.Arrow C
            X : CategoryTheory.CosimplicialObject.Augmented C
            G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
            x y : SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd x.len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp G.right (X.right.map ((SimplexCategor …
          -/
          rw [Category.assoc, ← X.right.map_comp]
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            F : CategoryTheory.Arrow C
            X : CategoryTheory.CosimplicialObject.Augmented C
            G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
            x y : SimplexCategory
            f : Quiver.Hom x y
            j✝ : Fin (HAdd.hAdd x.len 1)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp G.right (X.right.map ((SimplexCategor …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            F : CategoryTheory.Arrow C
            X : CategoryTheory.CosimplicialObject.Augmented C
            G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
            x y : SimplexCategory
            f : Quiver.Hom x y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.he …
          -/
        · dsimp
          simp only [Functor.const_obj_map, ← NatTrans.naturality, WidePushout.head_desc_assoc,
            WidePushout.head_desc, Category.assoc]
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
            F : CategoryTheory.Arrow C
            X : CategoryTheory.CosimplicialObject.Augmented C
            G : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
            x y : SimplexCategory
            f : Quiver.Hom x y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp G.left (X.hom.app y)) (CategoryTheory …
          -/
          erw [Category.id_comp] }
          /-
            🎉 no goals
          -/


/-- A helper function used in defining the Čech conerve adjunction. -/
@[simps]
def cechConerveEquiv (F : Arrow C) (X : CosimplicialObject.Augmented C) :
    (F.augmentedCechConerve ⟶ X) ≃ (F ⟶ Augmented.toArrow.obj X) where
  toFun := equivalenceLeftToRight _ _
  invFun := equivalenceRightToLeft _ _
  left_inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      F : CategoryTheory.Arrow C
      X : CategoryTheory.CosimplicialObject.Augmented C
      ⊢ Function.LeftInverse (CategoryTheory.CosimplicialObject.equivalenceRightToLe …
    -/
    intro A
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      F : CategoryTheory.Arrow C
      X : CategoryTheory.CosimplicialObject.Augmented C
      A : Quiver.Hom F.augmentedCechConerve X
      ⊢ Eq (CategoryTheory.CosimplicialObject.equivalenceRightToLeft F X (CategoryTh …
    -/
    ext x : 2
      /-
        case h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F.augmentedCechConerve X
        ⊢ Eq (CategoryTheory.CosimplicialObject.equivalenceRightToLeft F X (CategoryTh …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h₂.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F.augmentedCechConerve X
        x : SimplexCategory
        ⊢ Eq ((CategoryTheory.CosimplicialObject.equivalenceRightToLeft F X (CategoryT …
      -/
    · refine WidePushout.hom_ext _ _ _ (fun j => ?_) ?_
        /-
          case h₂.h.refine_1
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          A : Quiver.Hom F.augmentedCechConerve X
          x : SimplexCategory
          j : Fin (HAdd.hAdd x.len 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.ι  …
        -/
      · dsimp
        simp only [Category.assoc, ← NatTrans.naturality A.right, Arrow.augmentedCechConerve_right,
          SimplexCategory.len_mk, Arrow.cechConerve_map, colimit.ι_desc,
          WidePushoutShape.mkCocone_ι_app, colimit.ι_desc_assoc]
        /-
          case h₂.h.refine_1
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          A : Quiver.Hom F.augmentedCechConerve X
          x : SimplexCategory
          j : Fin (HAdd.hAdd x.len 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.ι  …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case h₂.h.refine_2
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          A : Quiver.Hom F.augmentedCechConerve X
          x : SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.he …
        -/
      · dsimp
        /-
          case h₂.h.refine_2
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          A : Quiver.Hom F.augmentedCechConerve X
          x : SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.he …
        -/
        rw [colimit.ι_desc]
        /-
          case h₂.h.refine_2
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
          F : CategoryTheory.Arrow C
          X : CategoryTheory.CosimplicialObject.Augmented C
          A : Quiver.Hom F.augmentedCechConerve X
          x : SimplexCategory
          ⊢ Eq ((CategoryTheory.Limits.WidePushoutShape.mkCocone (CategoryTheory.Categor …
        -/
        exact congr_app A.w x
        /-
          🎉 no goals
        -/
  right_inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      F : CategoryTheory.Arrow C
      X : CategoryTheory.CosimplicialObject.Augmented C
      ⊢ Function.RightInverse (CategoryTheory.CosimplicialObject.equivalenceRightToL …
    -/
    intro A
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
      F : CategoryTheory.Arrow C
      X : CategoryTheory.CosimplicialObject.Augmented C
      A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
      ⊢ Eq (CategoryTheory.CosimplicialObject.equivalenceLeftToRight F X (CategoryTh …
    -/
    ext
      /-
        case h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq (CategoryTheory.CosimplicialObject.equivalenceLeftToRight F X (CategoryTh …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq (CategoryTheory.CosimplicialObject.equivalenceLeftToRight F X (CategoryTh …
      -/
    · dsimp
      /-
        case h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.ι  …
      -/
      rw [WidePushout.ι_desc]
      /-
        case h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp A.right (X.right.map ((SimplexCategor …
      -/
      nth_rw 2 [← Category.comp_id A.right]
      /-
        case h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp A.right (X.right.map ((SimplexCategor …
      -/
      congr 1
      /-
        case h₂.e_a
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq (X.right.map ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0)) (Ca …
      -/
      convert X.right.map_id _
      /-
        case h.e'_2.h.h.e'_8
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        ⊢ Eq ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0) (CategoryTheory.C …
      -/
      ext ⟨a, ha⟩
      /-
        case h.e'_2.h.h.e'_8.a.h.h.mk.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        a : Nat
        ha : LT.lt a (HAdd.hAdd (SimplexCategory.mk 0).len 1)
        ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom ((SimplexCategory.mk 0).const (SimplexC …
      -/
      change a < 1 at ha
      /-
        case h.e'_2.h.h.e'_8.a.h.h.mk.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        a : Nat
        ha : LT.lt a 1
        ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom ((SimplexCategory.mk 0).const (SimplexC …
      -/
      change 0 = a
      /-
        case h.e'_2.h.h.e'_8.a.h.h.mk.h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : ∀ (n : Nat) (f : CategoryTheory.Arrow C), CategoryTheory.Limits.HasWid …
        F : CategoryTheory.Arrow C
        X : CategoryTheory.CosimplicialObject.Augmented C
        A : Quiver.Hom F (CategoryTheory.CosimplicialObject.Augmented.toArrow.obj X)
        a : Nat
        ha : LT.lt a 1
        ⊢ Eq 0 a
      -/
      omega
      /-
        🎉 no goals
      -/


/-- The augmented Čech conerve construction is left adjoint to the `toArrow` functor. -/
abbrev cechConerveAdjunction : augmentedCechConerve ⊣ (Augmented.toArrow : _ ⥤ Arrow C) :=
  Adjunction.mkOfHomEquiv { homEquiv := cechConerveEquiv }


/-- Given an object `X : C`, the natural simplicial object sending `[n]` to `Xⁿ⁺¹`. -/
def cechNerveTerminalFrom {C : Type u} [Category.{v} C] [HasFiniteProducts C] (X : C) :
    SimplicialObject C where
  obj n := ∏ᶜ fun _ : Fin (n.unop.len + 1) => X
  map f := Limits.Pi.lift fun i => Limits.Pi.π _ (f.unop.toOrderHom i)


/-- The diagram `Option ι ⥤ C` sending `none` to the terminal object and `some j` to `X`. -/
def wideCospan (X : C) : WidePullbackShape ι ⥤ C :=
  WidePullbackShape.wideCospan (terminal C) (fun _ : ι => X) fun _ => terminal.from X


instance uniqueToWideCospanNone (X Y : C) : Unique (Y ⟶ (wideCospan ι X).obj none) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    ι : Type w
    X Y : C
    ⊢ Unique (Quiver.Hom Y ((CategoryTheory.CechNerveTerminalFrom.wideCospan ι X). …
  -/
  dsimp [wideCospan]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    ι : Type w
    X Y : C
    ⊢ Unique (Quiver.Hom Y (CategoryTheory.Limits.terminal C))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The product `Xᶥ` is the vertex of a limit cone on `wideCospan ι X`. -/
def wideCospan.limitCone [Finite ι] (X : C) : LimitCone (wideCospan ι X) where
  cone :=
    { pt := ∏ᶜ fun _ : ι => X
      π :=
        { app := fun X => Option.casesOn X (terminal.from _) fun i => limit.π _ ⟨i⟩
          naturality := fun i j f => by
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              inst✝² : CategoryTheory.Limits.HasTerminal C
              ι : Type w
              inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
              inst✝ : Finite ι
              X : C
              i j : CategoryTheory.Limits.WidePullbackShape ι
              f : Quiver.Hom i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
            -/
            cases f
              /-
                case id
                C : Type u
                inst✝³ : CategoryTheory.Category.{v, u} C
                inst✝² : CategoryTheory.Limits.HasTerminal C
                ι : Type w
                inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝ : Finite ι
                X : C
                i : CategoryTheory.Limits.WidePullbackShape ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
              -/
            · cases i
              /-
                case id.none
                C : Type u
                inst✝³ : CategoryTheory.Category.{v, u} C
                inst✝² : CategoryTheory.Limits.HasTerminal C
                ι : Type w
                inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝ : Finite ι
                X : C
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
              -/
              all_goals dsimp; simp
              /-
                🎉 no goals
              -/
              /-
                case term
                C : Type u
                inst✝³ : CategoryTheory.Category.{v, u} C
                inst✝² : CategoryTheory.Limits.HasTerminal C
                ι : Type w
                inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝ : Finite ι
                X : C
                j✝ : ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
              -/
            · simp only [Functor.const_obj_obj, Functor.const_obj_map, terminal.comp_from]
              /-
                case term
                C : Type u
                inst✝³ : CategoryTheory.Category.{v, u} C
                inst✝² : CategoryTheory.Limits.HasTerminal C
                ι : Type w
                inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝ : Finite ι
                X : C
                j✝ : ι
                ⊢ Eq (CategoryTheory.Limits.terminal.from (CategoryTheory.Limits.piObj fun x = …
              -/
              subsingleton } }
              /-
                🎉 no goals
              -/
  isLimit :=
    { lift := fun s => Limits.Pi.lift fun j => s.π.app (some j)
                                             /-
                                               C : Type u
                                               inst✝³ : CategoryTheory.Category.{v, u} C
                                               inst✝² : CategoryTheory.Limits.HasTerminal C
                                               ι : Type w
                                               inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                               inst✝ : Finite ι
                                               X : C
                                               s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
                                               j : CategoryTheory.Limits.WidePullbackShape ι
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.Pi.l …
                                             -/
      fac := fun s j => Option.casesOn j (by subsingleton) fun _ => limit.lift_π _ _
                                             /-
                                               🎉 no goals
                                             -/
      uniq := fun s f h => by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasTerminal C
          ι : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : Finite ι
          X : C
          s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
          f : Quiver.Hom s.pt { pt := CategoryTheory.Limits.piObj fun x => X, π := { app …
          h : ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq (CategoryTheory.Cate …
          ⊢ Eq f ((fun s => CategoryTheory.Limits.Pi.lift fun j => s.π.app (Option.some  …
        -/
        dsimp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasTerminal C
          ι : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : Finite ι
          X : C
          s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
          f : Quiver.Hom s.pt { pt := CategoryTheory.Limits.piObj fun x => X, π := { app …
          h : ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq (CategoryTheory.Cate …
          ⊢ Eq f (CategoryTheory.Limits.Pi.lift fun j => s.π.app (Option.some j))
        -/
        ext j
        /-
          case h
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasTerminal C
          ι : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : Finite ι
          X : C
          s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
          f : Quiver.Hom s.pt { pt := CategoryTheory.Limits.piObj fun x => X, π := { app …
          h : ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq (CategoryTheory.Cate …
          j : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Pi.π (fun x  …
        -/
        dsimp only [Limits.Pi.lift]
        /-
          case h
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasTerminal C
          ι : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : Finite ι
          X : C
          s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
          f : Quiver.Hom s.pt { pt := CategoryTheory.Limits.piObj fun x => X, π := { app …
          h : ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq (CategoryTheory.Cate …
          j : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Pi.π (fun x  …
        -/
        rw [limit.lift_π]
        /-
          case h
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasTerminal C
          ι : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : Finite ι
          X : C
          s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
          f : Quiver.Hom s.pt { pt := CategoryTheory.Limits.piObj fun x => X, π := { app …
          h : ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq (CategoryTheory.Cate …
          j : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Pi.π (fun x  …
        -/
        dsimp
        /-
          case h
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasTerminal C
          ι : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : Finite ι
          X : C
          s : CategoryTheory.Limits.Cone (CategoryTheory.CechNerveTerminalFrom.wideCospa …
          f : Quiver.Hom s.pt { pt := CategoryTheory.Limits.piObj fun x => X, π := { app …
          h : ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq (CategoryTheory.Cate …
          j : ι
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Pi.π (fun x  …
        -/
        rw [← h (some j)] }
        /-
          🎉 no goals
        -/


instance hasWidePullback [Finite ι] (X : C) :
    HasWidePullback (Arrow.mk (terminal.from X)).right
      (fun _ : ι => (Arrow.mk (terminal.from X)).left)
      (fun _ => (Arrow.mk (terminal.from X)).hom) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasTerminal C
    ι : Type w
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : Finite ι
    X : C
    ⊢ CategoryTheory.Limits.HasWidePullback (CategoryTheory.Arrow.mk (CategoryTheo …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasTerminal C
    ι : Type w
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : Finite ι
    X : C
    val✝ : Fintype ι
    ⊢ CategoryTheory.Limits.HasWidePullback (CategoryTheory.Arrow.mk (CategoryTheo …
  -/
  exact ⟨⟨wideCospan.limitCone ι X⟩⟩
  /-
    🎉 no goals
  -/

-- Porting note: added to make the following definitions work

instance hasWidePullback' [Finite ι] (X : C) :
    HasWidePullback (⊤_ C)
      (fun _ : ι => X)
      (fun _ => terminal.from X) :=
  hasWidePullback _ _

-- Porting note: added to make the following definitions work

instance hasLimit_wideCospan [Finite ι] (X : C) : HasLimit (wideCospan ι X) := hasWidePullback _ _

-- Porting note: added to ease the definition of `iso`

/-- the isomorphism to the product induced by the limit cone `wideCospan ι X` -/
def wideCospan.limitIsoPi [Finite ι] (X : C) :
    limit (wideCospan ι X) ≅ ∏ᶜ fun _ : ι => X :=
  (IsLimit.conePointUniqueUpToIso (limit.isLimit _)
    (wideCospan.limitCone ι X).2)

-- Porting note: added to ease the definition of `iso`

@[reassoc (attr := simp)]
lemma wideCospan.limitIsoPi_inv_comp_pi [Finite ι] (X : C) (j : ι) :
    (wideCospan.limitIsoPi ι X).inv ≫ WidePullback.π _ j = Pi.π _ j :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ _


@[reassoc (attr := simp)]
lemma wideCospan.limitIsoPi_hom_comp_pi [Finite ι] (X : C) (j : ι) :
    (wideCospan.limitIsoPi ι X).hom ≫ Pi.π _ j = WidePullback.π _ j := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasTerminal C
    ι : Type w
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : Finite ι
    X : C
    j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CechNerveTerminalFrom …
  -/
  rw [← wideCospan.limitIsoPi_inv_comp_pi, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


/-- Given an object `X : C`, the Čech nerve of the hom to the terminal object `X ⟶ ⊤_ C` is
naturally isomorphic to a simplicial object sending `[n]` to `Xⁿ⁺¹` (when `C` is `G-Set`, this is
`EG`, the universal cover of the classifying space of `G`. -/
def iso (X : C) : (Arrow.mk (terminal.from X)).cechNerve ≅ cechNerveTerminalFrom X :=
  NatIso.ofComponents (fun _ => wideCospan.limitIsoPi _ _) (fun {m n} f => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      ι : Type w
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      X : C
      m n : Opposite SimplexCategory
      f : Quiver.Hom m n
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Arrow.mk (CategoryTh …
    -/
    dsimp only [cechNerveTerminalFrom, Arrow.cechNerve]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      ι : Type w
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      X : C
      m n : Opposite SimplexCategory
      f : Quiver.Hom m n
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
    -/
    ext ⟨j⟩
    /-
      case h.mk
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      ι : Type w
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      X : C
      m n : Opposite SimplexCategory
      f : Quiver.Hom m n
      j : Nat
      isLt✝ : LT.lt j (HAdd.hAdd (Opposite.unop n).len 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc, limit.lift_π, Fan.mk_π_app]
    erw [wideCospan.limitIsoPi_hom_comp_pi,
      wideCospan.limitIsoPi_hom_comp_pi, limit.lift_π]
    /-
      case h.mk
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      ι : Type w
      inst✝ : CategoryTheory.Limits.HasFiniteProducts C
      X : C
      m n : Opposite SimplexCategory
      f : Quiver.Hom m n
      j : Nat
      isLt✝ : LT.lt j (HAdd.hAdd (Opposite.unop n).len 1)
      ⊢ Eq ((CategoryTheory.Limits.WidePullbackShape.mkCone (CategoryTheory.Limits.W …
    -/
    rfl)
    /-
      🎉 no goals
    -/


