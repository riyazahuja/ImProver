/-- If `G` is adjoint to `F` then `F.unop` is adjoint to `G.unop`. -/
@[simps]
def unop {F : Cᵒᵖ ⥤ Dᵒᵖ} {G : Dᵒᵖ ⥤ Cᵒᵖ} (h : G ⊣ F) : F.unop ⊣ G.unop where
  unit := NatTrans.unop h.counit
  counit := NatTrans.unop h.unit
  left_triangle_components _ := Quiver.Hom.op_inj (h.right_triangle_components _)
  right_triangle_components _ := Quiver.Hom.op_inj (h.left_triangle_components _)


@[deprecated (since := "2025-01-01")] alias adjointOfOpAdjointOp := unop

@[deprecated (since := "2025-01-01")] alias adjointUnopOfAdjointOp := unop

@[deprecated (since := "2025-01-01")] alias unopAdjointOfOpAdjoint := unop

@[deprecated (since := "2025-01-01")] alias unopAdjointUnopOfAdjoint := unop


/-- If `G` is adjoint to `F` then `F.op` is adjoint to `G.op`. -/
@[simps]
def op {F : C ⥤ D} {G : D ⥤ C} (h : G ⊣ F) : F.op ⊣ G.op where
  unit := NatTrans.op h.counit
  counit := NatTrans.op h.unit
                                                        /-
                                                          C : Type u₁
                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                          D : Type u₂
                                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                          F : CategoryTheory.Functor C D
                                                          G : CategoryTheory.Functor D C
                                                          h : CategoryTheory.Adjunction G F
                                                          x✝ : Opposite C
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.map ((CategoryTheory.NatTrans.o …
                                                        -/
  left_triangle_components _ := Quiver.Hom.unop_inj (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                         /-
                                                           C : Type u₁
                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                           D : Type u₂
                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                           F : CategoryTheory.Functor C D
                                                           G : CategoryTheory.Functor D C
                                                           h : CategoryTheory.Adjunction G F
                                                           x✝ : Opposite D
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.NatTrans.op h.counit …
                                                         -/
  right_triangle_components _ := Quiver.Hom.unop_inj (by simp)
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2025-01-01")] alias opAdjointOpOfAdjoint := op

@[deprecated (since := "2025-01-01")] alias adjointOpOfAdjointUnop := op

@[deprecated (since := "2025-01-01")] alias opAdjointOfUnopAdjoint := op

@[deprecated (since := "2025-01-01")] alias adjointOfUnopAdjointUnop := op


/-- If `F` and `F'` are both adjoint to `G`, there is a natural isomorphism
`F.op ⋙ coyoneda ≅ F'.op ⋙ coyoneda`.
We use this in combination with `fullyFaithfulCancelRight` to show left adjoints are unique.
-/
def leftAdjointsCoyonedaEquiv {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G) :
    F.op ⋙ coyoneda ≅ F'.op ⋙ coyoneda :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
  -/
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F F' : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj1 : CategoryTheory.Adjunction F G
      adj2 : CategoryTheory.Adjunction F' G
      X : Opposite C
      ⊢ ∀ {X_1 Y : D} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
    -/
  NatIso.ofComponents fun X =>
    /-
      🎉 no goals
    -/
  /-
    🎉 no goals
  -/
    NatIso.ofComponents fun Y =>
      ((adj1.homEquiv X.unop Y).trans (adj2.homEquiv X.unop Y).symm).toIso


/-- Given two adjunctions, if the right adjoints are naturally isomorphic, then so are the left
adjoints.

Note: it is generally better to use `Adjunction.natIsoEquiv`, see the file `Adjunction.Unique`.
The reason this definition still exists is that apparently `CategoryTheory.extendAlongYonedaYoneda`
uses its definitional properties (TODO: figure out a way to avoid this).
-/
def natIsoOfRightAdjointNatIso {F F' : C ⥤ D} {G G' : D ⥤ C}
    (adj1 : F ⊣ G) (adj2 : F' ⊣ G') (r : G ≅ G') : F ≅ F' :=
  NatIso.removeOp ((Coyoneda.fullyFaithful.whiskeringRight _).isoEquiv.symm
    (leftAdjointsCoyonedaEquiv adj2 (adj1.ofNatIsoRight r)))


/-- Given two adjunctions, if the left adjoints are naturally isomorphic, then so are the right
adjoints.

Note: it is generally better to use `Adjunction.natIsoEquiv`, see the file `Adjunction.Unique`.
-/
def natIsoOfLeftAdjointNatIso {F F' : C ⥤ D} {G G' : D ⥤ C}
    (adj1 : F ⊣ G) (adj2 : F' ⊣ G') (l : F ≅ F') : G ≅ G' :=
  NatIso.removeOp (natIsoOfRightAdjointNatIso (op adj2) (op adj1) (NatIso.op l))


