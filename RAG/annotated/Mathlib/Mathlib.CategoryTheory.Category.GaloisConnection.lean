/-- A galois connection between preorders induces an adjunction between the associated categories.
-/
def GaloisConnection.adjunction {l : X → Y} {u : Y → X} (gc : GaloisConnection l u) :
    gc.monotone_l.functor ⊣ gc.monotone_u.functor :=
  CategoryTheory.Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y =>
        { toFun := fun f => CategoryTheory.homOfLE (gc.le_u f.le)
          invFun := fun f => CategoryTheory.homOfLE (gc.l_le f.le)
                         /-
                           X✝ : Type u
                           Y✝ : Type v
                           inst✝¹ : Preorder X✝
                           inst✝ : Preorder Y✝
                           l : X✝ → Y✝
                           u : Y✝ → X✝
                           gc : GaloisConnection l u
                           X : X✝
                           Y : Y✝
                           ⊢ Function.LeftInverse (fun f => CategoryTheory.homOfLE ⋯) fun f => CategoryTh …
                         -/
          left_inv := by aesop_cat
                         /-
                           🎉 no goals
                         -/
                          /-
                            X✝ : Type u
                            Y✝ : Type v
                            inst✝¹ : Preorder X✝
                            inst✝ : Preorder Y✝
                            l : X✝ → Y✝
                            u : Y✝ → X✝
                            gc : GaloisConnection l u
                            X : X✝
                            Y : Y✝
                            ⊢ Function.RightInverse (fun f => CategoryTheory.homOfLE ⋯) fun f => CategoryT …
                          -/
          right_inv := by aesop_cat } }
                          /-
                            🎉 no goals
                          -/


/-- An adjunction between preorder categories induces a galois connection.
-/
theorem Adjunction.gc {L : X ⥤ Y} {R : Y ⥤ X} (adj : L ⊣ R) : GaloisConnection L.obj R.obj :=
  fun x y =>
  ⟨fun h => ((adj.homEquiv x y).toFun h.hom).le, fun h => ((adj.homEquiv x y).invFun h.hom).le⟩


