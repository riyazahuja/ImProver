/-- When we have an adjunction `G ⊣ F`, any commutative square where the left
map is of the form `G.map i` and the right map is `p` has an "adjoint" commutative
square whose left map is `i` and whose right map is `F.map p`. -/
theorem right_adjoint (sq : CommSq u (G.map i) p v) (adj : G ⊣ F) :
    CommSq (adj.homEquiv _ _ u) i (F.map p) (adj.homEquiv _ _ v) :=
  ⟨by
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      u : Quiver.Hom (G.obj A) X
      v : Quiver.Hom (G.obj B) Y
      sq : CategoryTheory.CommSq u (G.map i) p v
      adj : CategoryTheory.Adjunction G F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv A X) u) (F.map p)) (Ca …
    -/
    simp only [Adjunction.homEquiv_unit, assoc, ← F.map_comp, sq.w]
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      u : Quiver.Hom (G.obj A) X
      v : Quiver.Hom (G.obj B) Y
      sq : CategoryTheory.CommSq u (G.map i) p v
      adj : CategoryTheory.Adjunction G F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app A) (F.map (CategoryTheo …
    -/
    rw [F.map_comp, Adjunction.unit_naturality_assoc]⟩
    /-
      🎉 no goals
    -/


/-- The liftings of a commutative are in bijection with the liftings of its (right)
adjoint square. -/
def rightAdjointLiftStructEquiv : sq.LiftStruct ≃ (sq.right_adjoint adj).LiftStruct where
  toFun l :=
    { l := adj.homEquiv _ _ l.l
                     /-
                       C : Type u_1
                       D : Type u_2
                       inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
                       inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
                       G : CategoryTheory.Functor C D
                       F : CategoryTheory.Functor D C
                       A B : C
                       X Y : D
                       i : Quiver.Hom A B
                       p : Quiver.Hom X Y
                       u : Quiver.Hom (G.obj A) X
                       v : Quiver.Hom (G.obj B) Y
                       sq : CategoryTheory.CommSq u (G.map i) p v
                       adj : CategoryTheory.Adjunction G F
                       l : sq.LiftStruct
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp i ((adj.homEquiv B X) l.l)) ((adj.hom …
                     -/
      fac_left := by rw [← adj.homEquiv_naturality_left, l.fac_left]
                     /-
                       🎉 no goals
                     -/
                      /-
                        C : Type u_1
                        D : Type u_2
                        inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
                        inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
                        G : CategoryTheory.Functor C D
                        F : CategoryTheory.Functor D C
                        A B : C
                        X Y : D
                        i : Quiver.Hom A B
                        p : Quiver.Hom X Y
                        u : Quiver.Hom (G.obj A) X
                        v : Quiver.Hom (G.obj B) Y
                        sq : CategoryTheory.CommSq u (G.map i) p v
                        adj : CategoryTheory.Adjunction G F
                        l : sq.LiftStruct
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv B X) l.l) (F.map p)) ( …
                      -/
      fac_right := by rw [← Adjunction.homEquiv_naturality_right, l.fac_right] }
                      /-
                        🎉 no goals
                      -/
  invFun l :=
    { l := (adj.homEquiv _ _).symm l.l
      fac_left := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
          inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom (G.obj A) X
          v : Quiver.Hom (G.obj B) Y
          sq : CategoryTheory.CommSq u (G.map i) p v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map i) ((adj.homEquiv B X).symm l. …
        -/
        rw [← Adjunction.homEquiv_naturality_left_symm, l.fac_left]
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
          inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom (G.obj A) X
          v : Quiver.Hom (G.obj B) Y
          sq : CategoryTheory.CommSq u (G.map i) p v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq ((adj.homEquiv A X).symm ((adj.homEquiv A X) u)) u
        -/
        apply (adj.homEquiv _ _).left_inv
        /-
          🎉 no goals
        -/
      fac_right := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
          inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom (G.obj A) X
          v : Quiver.Hom (G.obj B) Y
          sq : CategoryTheory.CommSq u (G.map i) p v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv B X).symm l.l) p) v
        -/
        rw [← Adjunction.homEquiv_naturality_right_symm, l.fac_right]
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
          inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom (G.obj A) X
          v : Quiver.Hom (G.obj B) Y
          sq : CategoryTheory.CommSq u (G.map i) p v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq ((adj.homEquiv B Y).symm ((adj.homEquiv B Y) v)) v
        -/
        apply (adj.homEquiv _ _).left_inv }
        /-
          🎉 no goals
        -/
                 /-
                   C : Type u_1
                   D : Type u_2
                   inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
                   inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
                   G : CategoryTheory.Functor C D
                   F : CategoryTheory.Functor D C
                   A B : C
                   X Y : D
                   i : Quiver.Hom A B
                   p : Quiver.Hom X Y
                   u : Quiver.Hom (G.obj A) X
                   v : Quiver.Hom (G.obj B) Y
                   sq : CategoryTheory.CommSq u (G.map i) p v
                   adj : CategoryTheory.Adjunction G F
                   ⊢ Function.LeftInverse (fun l => { l := (adj.homEquiv B X).symm l.l, fac_left  …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    D : Type u_2
                    inst✝¹ : CategoryTheory.Category.{?u.2734, u_1} C
                    inst✝ : CategoryTheory.Category.{?u.2738, u_2} D
                    G : CategoryTheory.Functor C D
                    F : CategoryTheory.Functor D C
                    A B : C
                    X Y : D
                    i : Quiver.Hom A B
                    p : Quiver.Hom X Y
                    u : Quiver.Hom (G.obj A) X
                    v : Quiver.Hom (G.obj B) Y
                    sq : CategoryTheory.CommSq u (G.map i) p v
                    adj : CategoryTheory.Adjunction G F
                    ⊢ Function.RightInverse (fun l => { l := (adj.homEquiv B X).symm l.l, fac_left …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- A square has a lifting if and only if its (right) adjoint square has a lifting. -/
theorem right_adjoint_hasLift_iff : HasLift (sq.right_adjoint adj) ↔ HasLift sq := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom (G.obj A) X
    v : Quiver.Hom (G.obj B) Y
    sq : CategoryTheory.CommSq u (G.map i) p v
    adj : CategoryTheory.Adjunction G F
    ⊢ Iff ⋯.HasLift sq.HasLift
  -/
  simp only [HasLift.iff]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom (G.obj A) X
    v : Quiver.Hom (G.obj B) Y
    sq : CategoryTheory.CommSq u (G.map i) p v
    adj : CategoryTheory.Adjunction G F
    ⊢ Iff (Nonempty ⋯.LiftStruct) (Nonempty sq.LiftStruct)
  -/
  exact Equiv.nonempty_congr (sq.rightAdjointLiftStructEquiv adj).symm
  /-
    🎉 no goals
  -/


instance [HasLift sq] : HasLift (sq.right_adjoint adj) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom (G.obj A) X
    v : Quiver.Hom (G.obj B) Y
    sq : CategoryTheory.CommSq u (G.map i) p v
    adj : CategoryTheory.Adjunction G F
    inst✝ : sq.HasLift
    ⊢ ⋯.HasLift
  -/
  rw [right_adjoint_hasLift_iff]
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom (G.obj A) X
    v : Quiver.Hom (G.obj B) Y
    sq : CategoryTheory.CommSq u (G.map i) p v
    adj : CategoryTheory.Adjunction G F
    inst✝ : sq.HasLift
    ⊢ CategoryTheory.CommSq.HasLift ?sq
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- When we have an adjunction `G ⊣ F`, any commutative square where the left
map is of the form `i` and the right map is `F.map p` has an "adjoint" commutative
square whose left map is `G.map i` and whose right map is `p`. -/
theorem left_adjoint (sq : CommSq u i (F.map p) v) (adj : G ⊣ F) :
    CommSq ((adj.homEquiv _ _).symm u) (G.map i) p ((adj.homEquiv _ _).symm v) :=
  ⟨by
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      u : Quiver.Hom A (F.obj X)
      v : Quiver.Hom B (F.obj Y)
      sq : CategoryTheory.CommSq u i (F.map p) v
      adj : CategoryTheory.Adjunction G F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv A X).symm u) p) (Categ …
    -/
    simp only [Adjunction.homEquiv_counit, assoc, ← G.map_comp_assoc, ← sq.w]
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      u : Quiver.Hom A (F.obj X)
      v : Quiver.Hom B (F.obj Y)
      sq : CategoryTheory.CommSq u i (F.map p) v
      adj : CategoryTheory.Adjunction G F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map u) (CategoryTheory.CategoryStr …
    -/
    rw [G.map_comp, assoc, Adjunction.counit_naturality]⟩
    /-
      🎉 no goals
    -/


/-- The liftings of a commutative are in bijection with the liftings of its (left)
adjoint square. -/
def leftAdjointLiftStructEquiv :
    sq.LiftStruct ≃ (sq.left_adjoint adj).LiftStruct where
  toFun l :=
    { l := (adj.homEquiv _ _).symm l.l
                     /-
                       C : Type u_1
                       D : Type u_2
                       inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
                       inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
                       G : CategoryTheory.Functor C D
                       F : CategoryTheory.Functor D C
                       A B : C
                       X Y : D
                       i : Quiver.Hom A B
                       p : Quiver.Hom X Y
                       u : Quiver.Hom A (F.obj X)
                       v : Quiver.Hom B (F.obj Y)
                       sq : CategoryTheory.CommSq u i (F.map p) v
                       adj : CategoryTheory.Adjunction G F
                       l : sq.LiftStruct
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map i) ((adj.homEquiv B X).symm l. …
                     -/
      fac_left := by rw [← adj.homEquiv_naturality_left_symm, l.fac_left]
                     /-
                       🎉 no goals
                     -/
                      /-
                        C : Type u_1
                        D : Type u_2
                        inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
                        inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
                        G : CategoryTheory.Functor C D
                        F : CategoryTheory.Functor D C
                        A B : C
                        X Y : D
                        i : Quiver.Hom A B
                        p : Quiver.Hom X Y
                        u : Quiver.Hom A (F.obj X)
                        v : Quiver.Hom B (F.obj Y)
                        sq : CategoryTheory.CommSq u i (F.map p) v
                        adj : CategoryTheory.Adjunction G F
                        l : sq.LiftStruct
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv B X).symm l.l) p) ((ad …
                      -/
      fac_right := by rw [← adj.homEquiv_naturality_right_symm, l.fac_right] }
                      /-
                        🎉 no goals
                      -/
  invFun l :=
    { l := (adj.homEquiv _ _) l.l
      fac_left := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
          inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom A (F.obj X)
          v : Quiver.Hom B (F.obj Y)
          sq : CategoryTheory.CommSq u i (F.map p) v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq (CategoryTheory.CategoryStruct.comp i ((adj.homEquiv B X) l.l)) u
        -/
        rw [← adj.homEquiv_naturality_left, l.fac_left]
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
          inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom A (F.obj X)
          v : Quiver.Hom B (F.obj Y)
          sq : CategoryTheory.CommSq u i (F.map p) v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq ((adj.homEquiv A X) ((adj.homEquiv A X).symm u)) u
        -/
        apply (adj.homEquiv _ _).right_inv
        /-
          🎉 no goals
        -/
      fac_right := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
          inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom A (F.obj X)
          v : Quiver.Hom B (F.obj Y)
          sq : CategoryTheory.CommSq u i (F.map p) v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv B X) l.l) (F.map p)) v
        -/
        rw [← adj.homEquiv_naturality_right, l.fac_right]
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
          inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
          G : CategoryTheory.Functor C D
          F : CategoryTheory.Functor D C
          A B : C
          X Y : D
          i : Quiver.Hom A B
          p : Quiver.Hom X Y
          u : Quiver.Hom A (F.obj X)
          v : Quiver.Hom B (F.obj Y)
          sq : CategoryTheory.CommSq u i (F.map p) v
          adj : CategoryTheory.Adjunction G F
          l : ⋯.LiftStruct
          ⊢ Eq ((adj.homEquiv B Y) ((adj.homEquiv B Y).symm v)) v
        -/
        apply (adj.homEquiv _ _).right_inv }
        /-
          🎉 no goals
        -/
                 /-
                   C : Type u_1
                   D : Type u_2
                   inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
                   inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
                   G : CategoryTheory.Functor C D
                   F : CategoryTheory.Functor D C
                   A B : C
                   X Y : D
                   i : Quiver.Hom A B
                   p : Quiver.Hom X Y
                   u : Quiver.Hom A (F.obj X)
                   v : Quiver.Hom B (F.obj Y)
                   sq : CategoryTheory.CommSq u i (F.map p) v
                   adj : CategoryTheory.Adjunction G F
                   ⊢ Function.LeftInverse (fun l => { l := (adj.homEquiv B X) l.l, fac_left := ⋯, …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    D : Type u_2
                    inst✝¹ : CategoryTheory.Category.{?u.12545, u_1} C
                    inst✝ : CategoryTheory.Category.{?u.12549, u_2} D
                    G : CategoryTheory.Functor C D
                    F : CategoryTheory.Functor D C
                    A B : C
                    X Y : D
                    i : Quiver.Hom A B
                    p : Quiver.Hom X Y
                    u : Quiver.Hom A (F.obj X)
                    v : Quiver.Hom B (F.obj Y)
                    sq : CategoryTheory.CommSq u i (F.map p) v
                    adj : CategoryTheory.Adjunction G F
                    ⊢ Function.RightInverse (fun l => { l := (adj.homEquiv B X) l.l, fac_left := ⋯ …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- A (left) adjoint square has a lifting if and only if the original square has a lifting. -/
theorem left_adjoint_hasLift_iff : HasLift (sq.left_adjoint adj) ↔ HasLift sq := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom A (F.obj X)
    v : Quiver.Hom B (F.obj Y)
    sq : CategoryTheory.CommSq u i (F.map p) v
    adj : CategoryTheory.Adjunction G F
    ⊢ Iff ⋯.HasLift sq.HasLift
  -/
  simp only [HasLift.iff]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom A (F.obj X)
    v : Quiver.Hom B (F.obj Y)
    sq : CategoryTheory.CommSq u i (F.map p) v
    adj : CategoryTheory.Adjunction G F
    ⊢ Iff (Nonempty ⋯.LiftStruct) (Nonempty sq.LiftStruct)
  -/
  exact Equiv.nonempty_congr (sq.leftAdjointLiftStructEquiv adj).symm
  /-
    🎉 no goals
  -/


instance [HasLift sq] : HasLift (sq.left_adjoint adj) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom A (F.obj X)
    v : Quiver.Hom B (F.obj Y)
    sq : CategoryTheory.CommSq u i (F.map p) v
    adj : CategoryTheory.Adjunction G F
    inst✝ : sq.HasLift
    ⊢ ⋯.HasLift
  -/
  rw [left_adjoint_hasLift_iff]
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    u : Quiver.Hom A (F.obj X)
    v : Quiver.Hom B (F.obj Y)
    sq : CategoryTheory.CommSq u i (F.map p) v
    adj : CategoryTheory.Adjunction G F
    inst✝ : sq.HasLift
    ⊢ CategoryTheory.CommSq.HasLift ?sq
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem hasLiftingProperty_iff (adj : G ⊣ F) {A B : C} {X Y : D} (i : A ⟶ B) (p : X ⟶ Y) :
    HasLiftingProperty (G.map i) p ↔ HasLiftingProperty i (F.map p) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction G F
    A B : C
    X Y : D
    i : Quiver.Hom A B
    p : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.HasLiftingProperty (G.map i) p) (CategoryTheory.HasLifti …
  -/
  constructor <;> intro <;> constructor <;> intro f g sq
    /-
      case mp.sq_hasLift
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      a✝ : CategoryTheory.HasLiftingProperty (G.map i) p
      f : Quiver.Hom A (F.obj X)
      g : Quiver.Hom B (F.obj Y)
      sq : CategoryTheory.CommSq f i (F.map p) g
      ⊢ sq.HasLift
    -/
  · rw [← sq.left_adjoint_hasLift_iff adj]
    /-
      case mp.sq_hasLift
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      a✝ : CategoryTheory.HasLiftingProperty (G.map i) p
      f : Quiver.Hom A (F.obj X)
      g : Quiver.Hom B (F.obj Y)
      sq : CategoryTheory.CommSq f i (F.map p) g
      ⊢ ⋯.HasLift
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr.sq_hasLift
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      a✝ : CategoryTheory.HasLiftingProperty i (F.map p)
      f : Quiver.Hom (G.obj A) X
      g : Quiver.Hom (G.obj B) Y
      sq : CategoryTheory.CommSq f (G.map i) p g
      ⊢ sq.HasLift
    -/
  · rw [← sq.right_adjoint_hasLift_iff adj]
    /-
      case mpr.sq_hasLift
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction G F
      A B : C
      X Y : D
      i : Quiver.Hom A B
      p : Quiver.Hom X Y
      a✝ : CategoryTheory.HasLiftingProperty i (F.map p)
      f : Quiver.Hom (G.obj A) X
      g : Quiver.Hom (G.obj B) Y
      sq : CategoryTheory.CommSq f (G.map i) p g
      ⊢ ⋯.HasLift
    -/
    infer_instance
    /-
      🎉 no goals
    -/


