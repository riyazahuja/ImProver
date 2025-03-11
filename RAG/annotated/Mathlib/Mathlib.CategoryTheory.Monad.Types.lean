/-- A lawful `Control.Monad` gives a category theory `Monad` on the category of types.
-/
@[simps!]
def ofTypeMonad : Monad (Type u) where
  toFunctor := ofTypeFunctor m
  η := ⟨@pure m _, fun _ _ f => funext fun x => (LawfulApplicative.map_pure f x).symm⟩
                                                              /-
                                                                m : Type u → Type u
                                                                inst✝¹ : _root_.Monad m
                                                                inst✝ : LawfulMonad m
                                                                α β : Type u
                                                                f : α → β
                                                                a : ((CategoryTheory.ofTypeFunctor m).comp (CategoryTheory.ofTypeFunctor m)).o …
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.ofTypeFunctor m).co …
                                                              -/
  μ := ⟨@joinM m _, fun α β (f : α → β) => funext fun a => by apply joinM_map_map⟩
                                                              /-
                                                                🎉 no goals
                                                              -/
                                /-
                                  m : Type u → Type u
                                  inst✝¹ : _root_.Monad m
                                  inst✝ : LawfulMonad m
                                  α : Type u
                                  a : (CategoryTheory.ofTypeFunctor m).obj (((CategoryTheory.ofTypeFunctor m).co …
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeFunctor m).map …
                                -/
  assoc α := funext fun a => by apply joinM_map_joinM
                                /-
                                  🎉 no goals
                                -/
                                    /-
                                      m : Type u → Type u
                                      inst✝¹ : _root_.Monad m
                                      inst✝ : LawfulMonad m
                                      α : Type u
                                      a : (CategoryTheory.Functor.id (Type u)).obj ((CategoryTheory.ofTypeFunctor m) …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ app := @Pure.pure m Applicative.to …
                                    -/
  left_unit α := funext fun a => by apply joinM_pure
                                    /-
                                      🎉 no goals
                                    -/
                                     /-
                                       m : Type u → Type u
                                       inst✝¹ : _root_.Monad m
                                       inst✝ : LawfulMonad m
                                       α : Type u
                                       a : (CategoryTheory.ofTypeFunctor m).obj ((CategoryTheory.Functor.id (Type u)) …
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeFunctor m).map …
                                     -/
  right_unit α := funext fun a => by apply joinM_map_pure
                                     /-
                                       🎉 no goals
                                     -/


/-- The `Kleisli` category of a `Control.Monad` is equivalent to the `Kleisli` category of its
category-theoretic version, provided the monad is lawful.
-/
@[simps]
def eq : KleisliCat m ≌ Kleisli (ofTypeMonad m) where
  functor :=
    { obj := fun X => X
      map := fun f => f
      map_id := fun _ => rfl
      map_comp := fun f g => by
        --unfold_projs
        /-
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.KleisliCat m
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => f }.map (CategoryTheory.Categ …
        -/
        funext t
        -- Porting note: missing tactic `unfold_projs`, using `change` instead.
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.KleisliCat m
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => f }.map (CategoryTheory.Categ …
        -/
        change _ = joinM (g <$> (f t))
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.KleisliCat m
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => f }.map (CategoryTheory.Categ …
        -/
        simp only [joinM, seq_bind_eq, Function.id_comp]
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.KleisliCat m
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g t) (Bind.bind (f t) g)
        -/
        rfl }
        /-
          🎉 no goals
        -/
  inverse :=
    { obj := fun X => X
      map := fun f => f
      map_id := fun _ => rfl
      map_comp := fun f g => by
        --unfold_projs
        -- Porting note: Need these instances for some lemmas below.
        --Should they be added as actual instances elsewhere?
        letI : _root_.Monad (ofTypeMonad m).obj :=
          show _root_.Monad m from inferInstance
        letI : LawfulMonad (ofTypeMonad m).obj :=
          show LawfulMonad m from inferInstance
        /-
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.Kleisli (CategoryTheory.ofTypeMonad m)
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          this✝ : _root_.Monad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstanc …
          this : LawfulMonad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstance  …
          ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => f }.map (CategoryTheory.Categ …
        -/
        funext t
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.Kleisli (CategoryTheory.ofTypeMonad m)
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          this✝ : _root_.Monad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstanc …
          this : LawfulMonad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstance  …
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => f }.map (CategoryTheory.Categ …
        -/
        dsimp
        -- Porting note: missing tactic `unfold_projs`, using `change` instead.
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.Kleisli (CategoryTheory.ofTypeMonad m)
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          this✝ : _root_.Monad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstanc …
          this : LawfulMonad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstance  …
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g t) (CategoryTheory.CategoryStruct …
        -/
        change joinM (g <$> (f t)) = _
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.Kleisli (CategoryTheory.ofTypeMonad m)
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          this✝ : _root_.Monad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstanc …
          this : LawfulMonad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstance  …
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq (joinM (Functor.map g (f t))) (CategoryTheory.CategoryStruct.comp f g t)
        -/
        simp only [joinM, seq_bind_eq, Function.id_comp]
        /-
          case h
          m : Type u → Type u
          inst✝¹ : _root_.Monad m
          inst✝ : LawfulMonad m
          X✝ Y✝ Z✝ : CategoryTheory.Kleisli (CategoryTheory.ofTypeMonad m)
          f : Quiver.Hom X✝ Y✝
          g : Quiver.Hom Y✝ Z✝
          this✝ : _root_.Monad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstanc …
          this : LawfulMonad (CategoryTheory.ofTypeMonad m).obj := letFun inferInstance  …
          t : { obj := fun X => X, map := fun {X Y} f => f }.obj X✝
          ⊢ Eq (Bind.bind (f t) g) (CategoryTheory.CategoryStruct.comp f g t)
        -/
        rfl }
        /-
          🎉 no goals
        -/
  unitIso := by
    /-
      m : Type u → Type u
      inst✝¹ : _root_.Monad m
      inst✝ : LawfulMonad m
      ⊢ CategoryTheory.Iso (CategoryTheory.Functor.id (CategoryTheory.KleisliCat m)) …
    -/
    refine NatIso.ofComponents (fun X => Iso.refl X) fun f => ?_
    /-
      m : Type u → Type u
      inst✝¹ : _root_.Monad m
      inst✝ : LawfulMonad m
      X✝ Y✝ : CategoryTheory.KleisliCat m
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
    -/
    change f >=> pure = pure >=> f
    /-
      m : Type u → Type u
      inst✝¹ : _root_.Monad m
      inst✝ : LawfulMonad m
      X✝ Y✝ : CategoryTheory.KleisliCat m
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (Bind.kleisliRight f Pure.pure) (Bind.kleisliRight Pure.pure f)
    -/
    simp [functor_norm]
    /-
      🎉 no goals
    -/
               /-
                 m : Type u → Type u
                 inst✝¹ : _root_.Monad m
                 inst✝ : LawfulMonad m
                 ⊢ ∀ {X Y : CategoryTheory.Kleisli (CategoryTheory.ofTypeMonad m)} (f : Quiver. …
               -/
  counitIso := NatIso.ofComponents fun X => Iso.refl X
               /-
                 🎉 no goals
               -/


