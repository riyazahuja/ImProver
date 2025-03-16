/--
A functor `F : C ⥤ D` is final if for every `d : D`, the comma category of morphisms `d ⟶ F.obj c`
is connected.

See <https://stacks.math.columbia.edu/tag/04E6>
-/
class Final (F : C ⥤ D) : Prop where
  out (d : D) : IsConnected (StructuredArrow d F)


/-- A functor `F : C ⥤ D` is initial if for every `d : D`, the comma category of morphisms
`F.obj c ⟶ d` is connected.
-/
class Initial (F : C ⥤ D) : Prop where
  out (d : D) : IsConnected (CostructuredArrow F d)


instance final_op_of_initial (F : C ⥤ D) [Initial F] : Final F.op where
  out d := isConnected_of_equivalent (costructuredArrowOpEquivalence F (unop d))


instance initial_op_of_final (F : C ⥤ D) [Final F] : Initial F.op where
  out d := isConnected_of_equivalent (structuredArrowOpEquivalence F (unop d))


theorem final_of_initial_op (F : C ⥤ D) [Initial F.op] : Final F :=
  {
    out := fun d =>
      @isConnected_of_isConnected_op _ _
        (isConnected_of_equivalent (structuredArrowOpEquivalence F d).symm) }


theorem initial_of_final_op (F : C ⥤ D) [Final F.op] : Initial F :=
  {
    out := fun d =>
      @isConnected_of_isConnected_op _ _
        (isConnected_of_equivalent (costructuredArrowOpEquivalence F d).symm) }


/-- If a functor `R : D ⥤ C` is a right adjoint, it is final. -/
theorem final_of_adjunction {L : C ⥤ D} {R : D ⥤ C} (adj : L ⊣ R) : Final R :=
  { out := fun c =>
      let u : StructuredArrow c R := StructuredArrow.mk (adj.unit.app c)
      @zigzag_isConnected _ _ ⟨u⟩ fun f g =>
        Relation.ReflTransGen.trans
          (Relation.ReflTransGen.single
            (show Zag f u from
                                                                                      /-
                                                                                        C : Type u₁
                                                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                        D : Type u₂
                                                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                        L : CategoryTheory.Functor C D
                                                                                        R : CategoryTheory.Functor D C
                                                                                        adj : CategoryTheory.Adjunction L R
                                                                                        c : C
                                                                                        u : CategoryTheory.StructuredArrow c R := CategoryTheory.StructuredArrow.mk (a …
                                                                                        f g : CategoryTheory.StructuredArrow c R
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp u.hom (R.map ((adj.homEquiv c f.right …
                                                                                      -/
              Or.inr ⟨StructuredArrow.homMk ((adj.homEquiv c f.right).symm f.hom) (by simp [u])⟩))
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
          (Relation.ReflTransGen.single
            (show Zag u g from
                                                                                      /-
                                                                                        C : Type u₁
                                                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                        D : Type u₂
                                                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                        L : CategoryTheory.Functor C D
                                                                                        R : CategoryTheory.Functor D C
                                                                                        adj : CategoryTheory.Adjunction L R
                                                                                        c : C
                                                                                        u : CategoryTheory.StructuredArrow c R := CategoryTheory.StructuredArrow.mk (a …
                                                                                        f g : CategoryTheory.StructuredArrow c R
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp u.hom (R.map ((adj.homEquiv c g.right …
                                                                                      -/
              Or.inl ⟨StructuredArrow.homMk ((adj.homEquiv c g.right).symm g.hom) (by simp [u])⟩)) }
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- If a functor `L : C ⥤ D` is a left adjoint, it is initial. -/
theorem initial_of_adjunction {L : C ⥤ D} {R : D ⥤ C} (adj : L ⊣ R) : Initial L :=
  { out := fun d =>
      let u : CostructuredArrow L d := CostructuredArrow.mk (adj.counit.app d)
      @zigzag_isConnected _ _ ⟨u⟩ fun f g =>
        Relation.ReflTransGen.trans
          (Relation.ReflTransGen.single
            (show Zag f u from
                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                  D : Type u₂
                                                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                  L : CategoryTheory.Functor C D
                                                                                  R : CategoryTheory.Functor D C
                                                                                  adj : CategoryTheory.Adjunction L R
                                                                                  d : D
                                                                                  u : CategoryTheory.CostructuredArrow L d := CategoryTheory.CostructuredArrow.m …
                                                                                  f g : CategoryTheory.CostructuredArrow L d
                                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map ((adj.homEquiv f.left d) f.hom …
                                                                                -/
              Or.inl ⟨CostructuredArrow.homMk (adj.homEquiv f.left d f.hom) (by simp [u])⟩))
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
          (Relation.ReflTransGen.single
            (show Zag u g from
                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                  D : Type u₂
                                                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                  L : CategoryTheory.Functor C D
                                                                                  R : CategoryTheory.Functor D C
                                                                                  adj : CategoryTheory.Adjunction L R
                                                                                  d : D
                                                                                  u : CategoryTheory.CostructuredArrow L d := CategoryTheory.CostructuredArrow.m …
                                                                                  f g : CategoryTheory.CostructuredArrow L d
                                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map ((adj.homEquiv g.left d) g.hom …
                                                                                -/
              Or.inr ⟨CostructuredArrow.homMk (adj.homEquiv g.left d g.hom) (by simp [u])⟩)) }
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance (priority := 100) final_of_isRightAdjoint (F : C ⥤ D) [IsRightAdjoint F] : Final F :=
  final_of_adjunction (Adjunction.ofIsRightAdjoint F)


instance (priority := 100) initial_of_isLeftAdjoint (F : C ⥤ D) [IsLeftAdjoint F] : Initial F :=
  initial_of_adjunction (Adjunction.ofIsLeftAdjoint F)


theorem final_of_natIso {F F' : C ⥤ D} [Final F] (i : F ≅ F') : Final F' where
  out _ := isConnected_of_equivalent (StructuredArrow.mapNatIso i)


theorem final_natIso_iff {F F' : C ⥤ D} (i : F ≅ F') : Final F ↔ Final F' :=
  ⟨fun _ => final_of_natIso i, fun _ => final_of_natIso i.symm⟩


theorem initial_of_natIso {F F' : C ⥤ D} [Initial F] (i : F ≅ F') : Initial F' where
  out _ := isConnected_of_equivalent (CostructuredArrow.mapNatIso i)


theorem initial_natIso_iff {F F' : C ⥤ D} (i : F ≅ F') : Initial F ↔ Initial F' :=
  ⟨fun _ => initial_of_natIso i, fun _ => initial_of_natIso i.symm⟩


instance (d : D) : Nonempty (StructuredArrow d F) :=
  IsConnected.is_nonempty


/--
When `F : C ⥤ D` is final, we denote by `lift F d` an arbitrary choice of object in `C` such that
there exists a morphism `d ⟶ F.obj (lift F d)`.
-/
def lift (d : D) : C :=
  (Classical.arbitrary (StructuredArrow d F)).right


/-- When `F : C ⥤ D` is final, we denote by `homToLift` an arbitrary choice of morphism
`d ⟶ F.obj (lift F d)`.
-/
def homToLift (d : D) : d ⟶ F.obj (lift F d) :=
  (Classical.arbitrary (StructuredArrow d F)).hom


/-- We provide an induction principle for reasoning about `lift` and `homToLift`.
We want to perform some construction (usually just a proof) about
the particular choices `lift F d` and `homToLift F d`,
it suffices to perform that construction for some other pair of choices
(denoted `X₀ : C` and `k₀ : d ⟶ F.obj X₀` below),
and to show how to transport such a construction
*both* directions along a morphism between such choices.
-/
def induction {d : D} (Z : ∀ (X : C) (_ : d ⟶ F.obj X), Sort*)
    (h₁ :
      ∀ (X₁ X₂) (k₁ : d ⟶ F.obj X₁) (k₂ : d ⟶ F.obj X₂) (f : X₁ ⟶ X₂),
        k₁ ≫ F.map f = k₂ → Z X₁ k₁ → Z X₂ k₂)
    (h₂ :
      ∀ (X₁ X₂) (k₁ : d ⟶ F.obj X₁) (k₂ : d ⟶ F.obj X₂) (f : X₁ ⟶ X₂),
        k₁ ≫ F.map f = k₂ → Z X₂ k₂ → Z X₁ k₁)
    {X₀ : C} {k₀ : d ⟶ F.obj X₀} (z : Z X₀ k₀) : Z (lift F d) (homToLift F d) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Final
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    d : D
    Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
    h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
    h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
    X₀ : C
    k₀ : Quiver.Hom d (F.obj X₀)
    z : Z X₀ k₀
    ⊢ Z (CategoryTheory.Functor.Final.lift F d) (CategoryTheory.Functor.Final.homT …
  -/
  apply Nonempty.some
  apply
    @isPreconnected_induction _ _ _ (fun Y : StructuredArrow d F => Z Y.right Y.hom) _ _
      (StructuredArrow.mk k₀) z
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      ⊢ {j₁ j₂ : CategoryTheory.StructuredArrow d F} → Quiver.Hom j₁ j₂ → (fun Y =>  …
    -/
  · intro j₁ j₂ f a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₁.right j₁.hom
      ⊢ Z j₂.right j₂.hom
    -/
    fapply h₁ _ _ _ _ f.right _ a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₁.right j₁.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp j₁.hom (F.map f.right)) j₂.hom
    -/
    convert f.w.symm
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₁.right j₁.hom
      e_1✝ : Eq (Quiver.Hom d (F.obj j₂.right)) (Quiver.Hom ((CategoryTheory.Functor …
      ⊢ Eq j₂.hom (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.fromP …
    -/
    dsimp
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₁.right j₁.hom
      e_1✝ : Eq (Quiver.Hom d (F.obj j₂.right)) (Quiver.Hom ((CategoryTheory.Functor …
      ⊢ Eq j₂.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      ⊢ {j₁ j₂ : CategoryTheory.StructuredArrow d F} → Quiver.Hom j₁ j₂ → (fun Y =>  …
    -/
  · intro j₁ j₂ f a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₂.right j₂.hom
      ⊢ Z j₁.right j₁.hom
    -/
    fapply h₂ _ _ _ _ f.right _ a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₂.right j₂.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp j₁.hom (F.map f.right)) j₂.hom
    -/
    convert f.w.symm
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₂.right j₂.hom
      e_1✝ : Eq (Quiver.Hom d (F.obj j₂.right)) (Quiver.Hom ((CategoryTheory.Functor …
      ⊢ Eq j₂.hom (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.fromP …
    -/
    dsimp
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom d (F.obj X) → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom d (F.obj X₁)) → (k₂ : Quiver.Hom d (F.obj  …
      X₀ : C
      k₀ : Quiver.Hom d (F.obj X₀)
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.StructuredArrow d F
      f : Quiver.Hom j₁ j₂
      a : Z j₂.right j₂.hom
      e_1✝ : Eq (Quiver.Hom d (F.obj j₂.right)) (Quiver.Hom ((CategoryTheory.Functor …
      ⊢ Eq j₂.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given a cocone over `F ⋙ G`, we can construct a `Cocone G` with the same cocone point.
-/
@[simps]
def extendCocone : Cocone (F ⋙ G) ⥤ Cocone G where
  obj c :=
    { pt := c.pt
      ι :=
        { app := fun X => G.map (homToLift F X) ≫ c.ι.app (lift F X)
          naturality := fun X Y f => by
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.Final
              E : Type u₃
              inst✝ : CategoryTheory.Category.{v₃, u₃} E
              G : CategoryTheory.Functor D E
              c : CategoryTheory.Limits.Cocone (F.comp G)
              X Y : D
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => CategoryTheory.C …
            -/
            dsimp; simp only [Category.comp_id]
            -- This would be true if we'd chosen `lift F X` to be `lift F Y`
            -- and `homToLift F X` to be `f ≫ homToLift F Y`.
            apply
              induction F fun Z k =>
                G.map f ≫ G.map (homToLift F Y) ≫ c.ι.app (lift F Y) = G.map k ≫ c.ι.app Z
              /-
                case h₁
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Final
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cocone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom X (F.obj X₁)) (k₂ : Quiver.Hom X (F.obj X₂))  …
              -/
            · intro Z₁ Z₂ k₁ k₂ g a z
              /-
                case h₁
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Final
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cocone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                Z₁ Z₂ : C
                k₁ : Quiver.Hom X (F.obj Z₁)
                k₂ : Quiver.Hom X (F.obj Z₂)
                g : Quiver.Hom Z₁ Z₂
                a : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map g)) k₂
                z : Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryS …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryStr …
              -/
              rw [← a, Functor.map_comp, Category.assoc, ← Functor.comp_map, c.w, z]
              /-
                🎉 no goals
              -/
              /-
                case h₂
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Final
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cocone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom X (F.obj X₁)) (k₂ : Quiver.Hom X (F.obj X₂))  …
              -/
            · intro Z₁ Z₂ k₁ k₂ g a z
              /-
                case h₂
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Final
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cocone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                Z₁ Z₂ : C
                k₁ : Quiver.Hom X (F.obj Z₁)
                k₂ : Quiver.Hom X (F.obj Z₂)
                g : Quiver.Hom Z₁ Z₂
                a : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map g)) k₂
                z : Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryS …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryStr …
              -/
              rw [← a, Functor.map_comp, Category.assoc, ← Functor.comp_map, c.w] at z
              /-
                case h₂
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Final
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cocone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                Z₁ Z₂ : C
                k₁ : Quiver.Hom X (F.obj Z₁)
                k₂ : Quiver.Hom X (F.obj Z₂)
                g : Quiver.Hom Z₁ Z₂
                a : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map g)) k₂
                z : Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryS …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryStr …
              -/
              rw [z]
              /-
                🎉 no goals
              -/
              /-
                case z
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Final
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cocone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.CategoryStr …
              -/
            · rw [← Functor.map_comp_assoc] } }
              /-
                🎉 no goals
              -/
  map f := { hom := f.hom }


/-- Alternative equational lemma for `(extendCocone c).ι.app` in case a lift of the object
is given explicitly. -/
lemma extendCocone_obj_ι_app' (c : Cocone (F ⋙ G)) {X : D} {Y : C} (f : X ⟶ F.obj Y) :
    (extendCocone.obj c).ι.app X = G.map f ≫ c.ι.app Y := by
  apply induction (k₀ := f) (z := rfl) F fun Z g =>
    G.map g ≫ c.ι.app Z = G.map f ≫ c.ι.app Y
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cocone (F.comp G)
      X : D
      Y : C
      f : Quiver.Hom X (F.obj Y)
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom X (F.obj X₁)) (k₂ : Quiver.Hom X (F.obj X₂))  …
    -/
  · intro _ _ _ _ _ h₁ h₂
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cocone (F.comp G)
      X : D
      Y : C
      f : Quiver.Hom X (F.obj Y)
      X₁✝ X₂✝ : C
      k₁✝ : Quiver.Hom X (F.obj X₁✝)
      k₂✝ : Quiver.Hom X (F.obj X₂✝)
      f✝ : Quiver.Hom X₁✝ X₂✝
      h₁ : Eq (CategoryTheory.CategoryStruct.comp k₁✝ (F.map f✝)) k₂✝
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (G.map k₁✝) (c.ι.app X₁✝)) (Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map k₂✝) (c.ι.app X₂✝)) (CategoryT …
    -/
    simp [← h₁, ← Functor.comp_map, c.ι.naturality, h₂]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cocone (F.comp G)
      X : D
      Y : C
      f : Quiver.Hom X (F.obj Y)
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom X (F.obj X₁)) (k₂ : Quiver.Hom X (F.obj X₂))  …
    -/
  · intro _ _ _ _ _ h₁ h₂
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cocone (F.comp G)
      X : D
      Y : C
      f : Quiver.Hom X (F.obj Y)
      X₁✝ X₂✝ : C
      k₁✝ : Quiver.Hom X (F.obj X₁✝)
      k₂✝ : Quiver.Hom X (F.obj X₂✝)
      f✝ : Quiver.Hom X₁✝ X₂✝
      h₁ : Eq (CategoryTheory.CategoryStruct.comp k₁✝ (F.map f✝)) k₂✝
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (G.map k₂✝) (c.ι.app X₂✝)) (Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map k₁✝) (c.ι.app X₁✝)) (CategoryT …
    -/
    simp [← h₂, ← h₁, ← Functor.comp_map, c.ι.naturality]
    /-
      🎉 no goals
    -/


@[simp]
theorem colimit_cocone_comp_aux (s : Cocone (F ⋙ G)) (j : C) :
    G.map (homToLift F (F.obj j)) ≫ s.ι.app (lift F (F.obj j)) = s.ι.app j := by
  -- This point is that this would be true if we took `lift (F.obj j)` to just be `j`
  -- and `homToLift (F.obj j)` to be `𝟙 (F.obj j)`.
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Final
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    s : CategoryTheory.Limits.Cocone (F.comp G)
    j : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Functor.Final. …
  -/
  apply induction F fun X k => G.map k ≫ s.ι.app X = (s.ι.app j : _)
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j : C
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj j) (F.obj X₁)) (k₂ : Quiver.Hom (F.obj …
    -/
  · intro j₁ j₂ k₁ k₂ f w h
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j) (F.obj j₁)
      k₂ : Quiver.Hom (F.obj j) (F.obj j₂)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map f)) k₂
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map k₁) (s.ι.app j₁)) (s.ι.app j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map k₂) (s.ι.app j₂)) (s.ι.app j)
    -/
    rw [← w]
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j) (F.obj j₁)
      k₂ : Quiver.Hom (F.obj j) (F.obj j₂)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map f)) k₂
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map k₁) (s.ι.app j₁)) (s.ι.app j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
    -/
    rw [← s.w f] at h
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j) (F.obj j₁)
      k₂ : Quiver.Hom (F.obj j) (F.obj j₂)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map f)) k₂
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map k₁) (CategoryTheory.Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStruct …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j : C
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj j) (F.obj X₁)) (k₂ : Quiver.Hom (F.obj …
    -/
  · intro j₁ j₂ k₁ k₂ f w h
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j) (F.obj j₁)
      k₂ : Quiver.Hom (F.obj j) (F.obj j₂)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map f)) k₂
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map k₂) (s.ι.app j₂)) (s.ι.app j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map k₁) (s.ι.app j₁)) (s.ι.app j)
    -/
    rw [← w] at h
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j) (F.obj j₁)
      k₂ : Quiver.Hom (F.obj j) (F.obj j₂)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map f)) k₂
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map k₁) (s.ι.app j₁)) (s.ι.app j)
    -/
    rw [← s.w f]
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j) (F.obj j₁)
      k₂ : Quiver.Hom (F.obj j) (F.obj j₂)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp k₁ (F.map f)) k₂
      h : Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map k₁) (CategoryTheory.CategorySt …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case z
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cocone (F.comp G)
      j : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map ?k₀) (s.ι.app ?X₀)) (s.ι.app j)
    -/
  · exact s.w (𝟙 _)
    /-
      🎉 no goals
    -/


/-- If `F` is final,
the category of cocones on `F ⋙ G` is equivalent to the category of cocones on `G`,
for any `G : D ⥤ E`.
-/
@[simps]
def coconesEquiv : Cocone (F ⋙ G) ≌ Cocone G where
  functor := extendCocone
  inverse := Cocones.whiskering F
                                          /-
                                            C : Type u₁
                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                            D : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                            F : CategoryTheory.Functor C D
                                            inst✝¹ : F.Final
                                            E : Type u₃
                                            inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                            G : CategoryTheory.Functor D E
                                            c : CategoryTheory.Limits.Cocone (F.comp G)
                                            ⊢ ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor. …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun c => Cocones.ext (Iso.refl _)
             /-
               🎉 no goals
             -/
                                            /-
                                              C : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                              D : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                              F : CategoryTheory.Functor C D
                                              inst✝¹ : F.Final
                                              E : Type u₃
                                              inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                              G : CategoryTheory.Functor D E
                                              c : CategoryTheory.Limits.Cocone G
                                              ⊢ ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Limits. …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents fun c => Cocones.ext (Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- When `F : C ⥤ D` is final, and `t : Cocone G` for some `G : D ⥤ E`,
`t.whisker F` is a colimit cocone exactly when `t` is.
-/
def isColimitWhiskerEquiv (t : Cocone G) : IsColimit (t.whisker F) ≃ IsColimit t :=
  IsColimit.ofCoconeEquiv (coconesEquiv F G).symm


/-- When `F` is final, and `t : Cocone (F ⋙ G)`,
`extendCocone.obj t` is a colimit cocone exactly when `t` is.
-/
def isColimitExtendCoconeEquiv (t : Cocone (F ⋙ G)) :
    IsColimit (extendCocone.obj t) ≃ IsColimit t :=
  IsColimit.ofCoconeEquiv (coconesEquiv F G)


/-- Given a colimit cocone over `G : D ⥤ E` we can construct a colimit cocone over `F ⋙ G`. -/
@[simps]
def colimitCoconeComp (t : ColimitCocone G) : ColimitCocone (F ⋙ G) where
  cocone := _
  isColimit := (isColimitWhiskerEquiv F _).symm t.isColimit


instance (priority := 100) comp_hasColimit [HasColimit G] : HasColimit (F ⋙ G) :=
  HasColimit.mk (colimitCoconeComp F (getColimitCocone G))


instance (priority := 100) comp_preservesColimit {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [PreservesColimit G H] : PreservesColimit (F ⋙ G) H where
  preserves {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesColimit G H
      c : CategoryTheory.Limits.Cocone (F.comp G)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit (H.mapCocone c))
    -/
    refine ⟨isColimitExtendCoconeEquiv (G := G ⋙ H) F (H.mapCocone c) ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesColimit G H
      c : CategoryTheory.Limits.Cocone (F.comp G)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Functor.Final.extendCocone.o …
    -/
    let hc' := isColimitOfPreserves H ((isColimitExtendCoconeEquiv F c).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesColimit G H
      c : CategoryTheory.Limits.Cocone (F.comp G)
      hc : CategoryTheory.Limits.IsColimit c
      hc' : CategoryTheory.Limits.IsColimit (H.mapCocone (CategoryTheory.Functor.Fin …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Functor.Final.extendCocone.o …
    -/
    exact IsColimit.ofIsoColimit hc' (Cocones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


instance (priority := 100) comp_reflectsColimit {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [ReflectsColimit G H] : ReflectsColimit (F ⋙ G) H where
  reflects {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsColimit G H
      c : CategoryTheory.Limits.Cocone (F.comp G)
      hc : CategoryTheory.Limits.IsColimit (H.mapCocone c)
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
    -/
    refine ⟨isColimitExtendCoconeEquiv F _ (isColimitOfReflects H ?_)⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsColimit G H
      c : CategoryTheory.Limits.Cocone (F.comp G)
      hc : CategoryTheory.Limits.IsColimit (H.mapCocone c)
      ⊢ CategoryTheory.Limits.IsColimit (H.mapCocone (CategoryTheory.Functor.Final.e …
    -/
    let hc' := (isColimitExtendCoconeEquiv (G := G ⋙ H) F _).symm hc
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsColimit G H
      c : CategoryTheory.Limits.Cocone (F.comp G)
      hc : CategoryTheory.Limits.IsColimit (H.mapCocone c)
      hc' : CategoryTheory.Limits.IsColimit (CategoryTheory.Functor.Final.extendCoco …
      ⊢ CategoryTheory.Limits.IsColimit (H.mapCocone (CategoryTheory.Functor.Final.e …
    -/
    exact IsColimit.ofIsoColimit hc' (Cocones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


instance (priority := 100) compCreatesColimit {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [CreatesColimit G H] : CreatesColimit (F ⋙ G) H where
  lifts {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit G H
      c : CategoryTheory.Limits.Cocone ((F.comp G).comp H)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.LiftableCocone (F.comp G) H c
    -/
    refine ⟨(liftColimit ((isColimitExtendCoconeEquiv F (G := G ⋙ H) _).symm hc)).whisker F, ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit G H
      c : CategoryTheory.Limits.Cocone ((F.comp G).comp H)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Iso (H.mapCocone (CategoryTheory.Limits.Cocone.whisker F (Cat …
    -/
    let i := liftedColimitMapsToOriginal ((isColimitExtendCoconeEquiv F (G := G ⋙ H) _).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit G H
      c : CategoryTheory.Limits.Cocone ((F.comp G).comp H)
      hc : CategoryTheory.Limits.IsColimit c
      i : CategoryTheory.Iso (H.mapCocone (CategoryTheory.liftColimit ((CategoryTheo …
      ⊢ CategoryTheory.Iso (H.mapCocone (CategoryTheory.Limits.Cocone.whisker F (Cat …
    -/
    exact (Cocones.whiskering F).mapIso i ≪≫ ((coconesEquiv F (G ⋙ H)).unitIso.app _).symm
    /-
      🎉 no goals
    -/


instance colimit_pre_isIso [HasColimit G] : IsIso (colimit.pre G F) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Final
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasColimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre G F)
  -/
  rw [colimit.pre_eq (colimitCoconeComp F (getColimitCocone G)) (getColimitCocone G)]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Final
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasColimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  erw [IsColimit.desc_self]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Final
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasColimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Final
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasColimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- When `F : C ⥤ D` is final, and `G : D ⥤ E` has a colimit, then `F ⋙ G` has a colimit also and
`colimit (F ⋙ G) ≅ colimit G`

https://stacks.math.columbia.edu/tag/04E7
-/
@[simps! (config := .lemmasOnly)]
def colimitIso [HasColimit G] : colimit (F ⋙ G) ≅ colimit G :=
  asIso (colimit.pre G F)


@[reassoc (attr := simp)]
theorem ι_colimitIso_hom [HasColimit G] (X : C) :
    colimit.ι (F ⋙ G) X ≫ (colimitIso F G).hom = colimit.ι G (F.obj X) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Final
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasColimit G
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  simp [colimitIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem ι_colimitIso_inv [HasColimit G] (X : C) :
    colimit.ι G (F.obj X) ≫ (colimitIso F G).inv = colimit.ι (F ⋙ G) X := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Final
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasColimit G
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι G (F …
  -/
  simp [colimitIso]
  /-
    🎉 no goals
  -/


/-- A pointfree version of `colimitIso`, stating that whiskering by `F` followed by taking the
colimit is isomorpic to taking the colimit on the codomain of `F`. -/
def colimIso [HasColimitsOfShape D E] [HasColimitsOfShape C E] :
    (whiskeringLeft _ _ _).obj F ⋙ colim ≅ colim (J := D) (C := E) :=
  NatIso.ofComponents (fun G => colimitIso F G) fun f => by
    simp only [comp_obj, whiskeringLeft_obj_obj, colim_obj, comp_map, whiskeringLeft_obj_map,
      colim_map, colimitIso_hom]
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape D E
      inst✝ : CategoryTheory.Limits.HasColimitsOfShape C E
      X✝ Y✝ : CategoryTheory.Functor D E
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap (Cate …
    -/
    ext
    simp only [comp_obj, ι_colimMap_assoc, whiskerLeft_app, colimit.ι_pre, colimit.ι_pre_assoc,
      ι_colimMap]


/-- Given a colimit cocone over `F ⋙ G` we can construct a colimit cocone over `G`. -/
@[simps]
def colimitCoconeOfComp (t : ColimitCocone (F ⋙ G)) : ColimitCocone G where
  cocone := extendCocone.obj t.cocone
  isColimit := (isColimitExtendCoconeEquiv F _).symm t.isColimit


/-- When `F` is final, and `F ⋙ G` has a colimit, then `G` has a colimit also.

We can't make this an instance, because `F` is not determined by the goal.
(Even if this weren't a problem, it would cause a loop with `comp_hasColimit`.)
-/
theorem hasColimit_of_comp [HasColimit (F ⋙ G)] : HasColimit G :=
  HasColimit.mk (colimitCoconeOfComp F (getColimitCocone (F ⋙ G)))


theorem preservesColimit_of_comp {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [PreservesColimit (F ⋙ G) H] : PreservesColimit G H where
  preserves {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit (H.mapCocone c))
    -/
    refine ⟨isColimitWhiskerEquiv F _ ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cocone.whisker F (H.m …
    -/
    let hc' := isColimitOfPreserves H ((isColimitWhiskerEquiv F _).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      hc' : CategoryTheory.Limits.IsColimit (H.mapCocone (CategoryTheory.Limits.Coco …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cocone.whisker F (H.m …
    -/
    exact IsColimit.ofIsoColimit hc' (Cocones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


theorem reflectsColimit_of_comp {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [ReflectsColimit (F ⋙ G) H] : ReflectsColimit G H where
  reflects {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit (H.mapCocone c)
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit c)
    -/
    refine ⟨isColimitWhiskerEquiv F _ (isColimitOfReflects H ?_)⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit (H.mapCocone c)
      ⊢ CategoryTheory.Limits.IsColimit (H.mapCocone (CategoryTheory.Limits.Cocone.w …
    -/
    let hc' := (isColimitWhiskerEquiv F _).symm hc
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit (H.mapCocone c)
      hc' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cocone.whisker F  …
      ⊢ CategoryTheory.Limits.IsColimit (H.mapCocone (CategoryTheory.Limits.Cocone.w …
    -/
    exact IsColimit.ofIsoColimit hc' (Cocones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


/-- If `F` is final and `F ⋙ G` creates colimits of `H`, then so does `G`. -/
def createsColimitOfComp {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [CreatesColimit (F ⋙ G) H] : CreatesColimit G H where
  reflects := (reflectsColimit_of_comp F).reflects
  lifts {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone (G.comp H)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.LiftableCocone G H c
    -/
    refine ⟨(extendCocone (F := F)).obj (liftColimit ((isColimitWhiskerEquiv F _).symm hc)), ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone (G.comp H)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Iso (H.mapCocone (CategoryTheory.Functor.Final.extendCocone.o …
    -/
    let i := liftedColimitMapsToOriginal (K := (F ⋙ G)) ((isColimitWhiskerEquiv F _).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone (G.comp H)
      hc : CategoryTheory.Limits.IsColimit c
      i : CategoryTheory.Iso (H.mapCocone (CategoryTheory.liftColimit ((CategoryTheo …
      ⊢ CategoryTheory.Iso (H.mapCocone (CategoryTheory.Functor.Final.extendCocone.o …
    -/
    refine ?_ ≪≫ ((extendCocone (F := F)).mapIso i) ≪≫ ((coconesEquiv F (G ⋙ H)).counitIso.app _)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Final
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesColimit (F.comp G) H
      c : CategoryTheory.Limits.Cocone (G.comp H)
      hc : CategoryTheory.Limits.IsColimit c
      i : CategoryTheory.Iso (H.mapCocone (CategoryTheory.liftColimit ((CategoryTheo …
      ⊢ CategoryTheory.Iso (H.mapCocone (CategoryTheory.Functor.Final.extendCocone.o …
    -/
    exact Cocones.ext (Iso.refl _)
    /-
      🎉 no goals
    -/


include F in
theorem hasColimitsOfShape_of_final [HasColimitsOfShape C E] : HasColimitsOfShape D E where
  has_colimit := fun _ => hasColimit_of_comp F


include F in
theorem preservesColimitsOfShape_of_final {B : Type u₄} [Category.{v₄} B] (H : E ⥤ B)
    [PreservesColimitsOfShape C H] : PreservesColimitsOfShape D H where
  preservesColimit := preservesColimit_of_comp F


include F in
theorem reflectsColimitsOfShape_of_final {B : Type u₄} [Category.{v₄} B] (H : E ⥤ B)
    [ReflectsColimitsOfShape C H] : ReflectsColimitsOfShape D H where
  reflectsColimit := reflectsColimit_of_comp F


include F in
/-- If `H` creates colimits of shape `C` and `F : C ⥤ D` is final, then `H` creates colimits of
shape `D`. -/
def createsColimitsOfShapeOfFinal {B : Type u₄} [Category.{v₄} B] (H : E ⥤ B)
    [CreatesColimitsOfShape C H] : CreatesColimitsOfShape D H where
  CreatesColimit := createsColimitOfComp F


theorem zigzag_of_eqvGen_quot_rel {F : C ⥤ D} {d : D} {f₁ f₂ : ΣX, d ⟶ F.obj X}
    (t : Relation.EqvGen (Types.Quot.Rel.{v, v} (F ⋙ coyoneda.obj (op d))) f₁ f₂) :
    Zigzag (StructuredArrow.mk f₁.2) (StructuredArrow.mk f₂.2) := by
  induction t with
  | rel x y r =>
    obtain ⟨f, w⟩ := r
    fconstructor
    swap
    · fconstructor
    left; fconstructor
    exact StructuredArrow.homMk f
  | refl => fconstructor
  | symm x y _ ih =>
    apply zigzag_symmetric
    exact ih
  | trans x y z _ _ ih₁ ih₂ =>
    apply Relation.ReflTransGen.trans
    · exact ih₁
    · exact ih₂


/-- If `colimit (F ⋙ coyoneda.obj (op d)) ≅ PUnit` for all `d : D`, then `F` is final.
-/
theorem final_of_colimit_comp_coyoneda_iso_pUnit
    (I : ∀ d, colimit (F ⋙ coyoneda.obj (op d)) ≅ PUnit) : Final F :=
  ⟨fun d => by
    have : Nonempty (StructuredArrow d F) := by
      have := (I d).inv PUnit.unit
      obtain ⟨j, y, rfl⟩ := Limits.Types.jointly_surjective'.{v, v} this
      exact ⟨StructuredArrow.mk y⟩
    /-
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow d F)
    -/
    apply zigzag_isConnected
    /-
      case h
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      ⊢ ∀ (j₁ j₂ : CategoryTheory.StructuredArrow d F), CategoryTheory.Zigzag j₁ j₂
    -/
    rintro ⟨⟨⟨⟩⟩, X₁, f₁⟩ ⟨⟨⟨⟩⟩, X₂, f₂⟩
    /-
      case h.mk.mk.unit.mk.mk.unit
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      X₁ : C
      f₁ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      X₂ : C
      f₂ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      ⊢ CategoryTheory.Zigzag { left := { as := PUnit.unit }, right := X₁, hom := f₁ …
    -/
    let y₁ := colimit.ι (F ⋙ coyoneda.obj (op d)) X₁ f₁
    /-
      case h.mk.mk.unit.mk.mk.unit
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      X₁ : C
      f₁ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      X₂ : C
      f₂ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      y₁ : CategoryTheory.Limits.colimit (F.comp (CategoryTheory.coyoneda.obj { unop …
      ⊢ CategoryTheory.Zigzag { left := { as := PUnit.unit }, right := X₁, hom := f₁ …
    -/
    let y₂ := colimit.ι (F ⋙ coyoneda.obj (op d)) X₂ f₂
    have e : y₁ = y₂ := by
      apply (I d).toEquiv.injective
      ext
    /-
      case h.mk.mk.unit.mk.mk.unit
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      X₁ : C
      f₁ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      X₂ : C
      f₂ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      y₁ : CategoryTheory.Limits.colimit (F.comp (CategoryTheory.coyoneda.obj { unop …
      y₂ : CategoryTheory.Limits.colimit (F.comp (CategoryTheory.coyoneda.obj { unop …
      e : Eq y₁ y₂
      ⊢ CategoryTheory.Zigzag { left := { as := PUnit.unit }, right := X₁, hom := f₁ …
    -/
    have t := Types.colimit_eq.{v, v} e
    /-
      case h.mk.mk.unit.mk.mk.unit
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      X₁ : C
      f₁ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      X₂ : C
      f₂ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      y₁ : CategoryTheory.Limits.colimit (F.comp (CategoryTheory.coyoneda.obj { unop …
      y₂ : CategoryTheory.Limits.colimit (F.comp (CategoryTheory.coyoneda.obj { unop …
      e : Eq y₁ y₂
      t : Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryThe …
      ⊢ CategoryTheory.Zigzag { left := { as := PUnit.unit }, right := X₁, hom := f₁ …
    -/
    clear e y₁ y₂
    /-
      case h.mk.mk.unit.mk.mk.unit
      C : Type v
      inst✝¹ : CategoryTheory.Category.{v, v} C
      D : Type u₁
      inst✝ : CategoryTheory.Category.{v, u₁} D
      F : CategoryTheory.Functor C D
      I : (d : D) → CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (Categ …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      X₁ : C
      f₁ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      X₂ : C
      f₂ : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj { as := PUnit.unit } …
      t : Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryThe …
      ⊢ CategoryTheory.Zigzag { left := { as := PUnit.unit }, right := X₁, hom := f₁ …
    -/
    exact Final.zigzag_of_eqvGen_quot_rel t⟩
    /-
      🎉 no goals
    -/


/-- A variant of `final_of_colimit_comp_coyoneda_iso_pUnit` where we bind the various claims
    about `colimit (F ⋙ coyoneda.obj (Opposite.op d))` for each `d : D` into a single claim about
    the presheaf `colimit (F ⋙ yoneda)`. -/
theorem final_of_isTerminal_colimit_comp_yoneda
    (h : IsTerminal (colimit (F ⋙ yoneda))) : Final F := by
  /-
    C : Type v
    inst✝¹ : CategoryTheory.Category.{v, v} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} D
    F : CategoryTheory.Functor C D
    h : CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (F.comp Ca …
    ⊢ F.Final
  -/
  refine final_of_colimit_comp_coyoneda_iso_pUnit _ (fun d => ?_)
  /-
    C : Type v
    inst✝¹ : CategoryTheory.Category.{v, v} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} D
    F : CategoryTheory.Functor C D
    h : CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (F.comp Ca …
    d : D
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (F.comp (CategoryTheory.co …
  -/
  refine Types.isTerminalEquivIsoPUnit _ ?_
  /-
    C : Type v
    inst✝¹ : CategoryTheory.Category.{v, v} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} D
    F : CategoryTheory.Functor C D
    h : CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (F.comp Ca …
    d : D
    ⊢ CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (F.comp (Cat …
  -/
  let b := IsTerminal.isTerminalObj ((evaluation _ _).obj (Opposite.op d)) _ h
  /-
    C : Type v
    inst✝¹ : CategoryTheory.Category.{v, v} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v, u₁} D
    F : CategoryTheory.Functor C D
    h : CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (F.comp Ca …
    d : D
    b : CategoryTheory.Limits.IsTerminal (((CategoryTheory.evaluation (Opposite D) …
    ⊢ CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit (F.comp (Cat …
  -/
  exact b.ofIso <| preservesColimitIso ((evaluation _ _).obj (Opposite.op d)) (F ⋙ yoneda)
  /-
    🎉 no goals
  -/


/-- If the universal morphism `colimit (F ⋙ coyoneda.obj (op d)) ⟶ colimit (coyoneda.obj (op d))`
is an isomorphism (as it always is when `F` is final),
then `colimit (F ⋙ coyoneda.obj (op d)) ≅ PUnit`
(simply because `colimit (coyoneda.obj (op d)) ≅ PUnit`).
-/
def Final.colimitCompCoyonedaIso (d : D) [IsIso (colimit.pre (coyoneda.obj (op d)) F)] :
    colimit (F ⋙ coyoneda.obj (op d)) ≅ PUnit :=
  asIso (colimit.pre (coyoneda.obj (op d)) F) ≪≫ Coyoneda.colimitCoyonedaIso (op d)


theorem final_iff_isIso_colimit_pre : Final F ↔ ∀ G : D ⥤ Type v, IsIso (colimit.pre G F) :=
  ⟨fun _ => inferInstance,
   fun _ => final_of_colimit_comp_coyoneda_iso_pUnit _ fun _ => Final.colimitCompCoyonedaIso _ _⟩


instance (d : D) : Nonempty (CostructuredArrow F d) :=
  IsConnected.is_nonempty


/--
When `F : C ⥤ D` is initial, we denote by `lift F d` an arbitrary choice of object in `C` such that
there exists a morphism `F.obj (lift F d) ⟶ d`.
-/
def lift (d : D) : C :=
  (Classical.arbitrary (CostructuredArrow F d)).left


/-- When `F : C ⥤ D` is initial, we denote by `homToLift` an arbitrary choice of morphism
`F.obj (lift F d) ⟶ d`.
-/
def homToLift (d : D) : F.obj (lift F d) ⟶ d :=
  (Classical.arbitrary (CostructuredArrow F d)).hom


/-- We provide an induction principle for reasoning about `lift` and `homToLift`.
We want to perform some construction (usually just a proof) about
the particular choices `lift F d` and `homToLift F d`,
it suffices to perform that construction for some other pair of choices
(denoted `X₀ : C` and `k₀ : F.obj X₀ ⟶ d` below),
and to show how to transport such a construction
*both* directions along a morphism between such choices.
-/
def induction {d : D} (Z : ∀ (X : C) (_ : F.obj X ⟶ d), Sort*)
    (h₁ :
      ∀ (X₁ X₂) (k₁ : F.obj X₁ ⟶ d) (k₂ : F.obj X₂ ⟶ d) (f : X₁ ⟶ X₂),
        F.map f ≫ k₂ = k₁ → Z X₁ k₁ → Z X₂ k₂)
    (h₂ :
      ∀ (X₁ X₂) (k₁ : F.obj X₁ ⟶ d) (k₂ : F.obj X₂ ⟶ d) (f : X₁ ⟶ X₂),
        F.map f ≫ k₂ = k₁ → Z X₂ k₂ → Z X₁ k₁)
    {X₀ : C} {k₀ : F.obj X₀ ⟶ d} (z : Z X₀ k₀) : Z (lift F d) (homToLift F d) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Initial
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    d : D
    Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
    h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
    h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
    X₀ : C
    k₀ : Quiver.Hom (F.obj X₀) d
    z : Z X₀ k₀
    ⊢ Z (CategoryTheory.Functor.Initial.lift F d) (CategoryTheory.Functor.Initial. …
  -/
  apply Nonempty.some
  apply
    @isPreconnected_induction _ _ _ (fun Y : CostructuredArrow F d => Z Y.left Y.hom) _ _
      (CostructuredArrow.mk k₀) z
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      ⊢ {j₁ j₂ : CategoryTheory.CostructuredArrow F d} → Quiver.Hom j₁ j₂ → (fun Y = …
    -/
  · intro j₁ j₂ f a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₁.left j₁.hom
      ⊢ Z j₂.left j₂.hom
    -/
    fapply h₁ _ _ _ _ f.left _ a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₁.left j₁.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.left) j₂.hom) j₁.hom
    -/
    convert f.w
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₁.left j₁.hom
      e_1✝ : Eq (Quiver.Hom (F.obj j₁.left) d) (Quiver.Hom (F.obj j₁.left) ((Categor …
      ⊢ Eq j₁.hom (CategoryTheory.CategoryStruct.comp j₁.hom ((CategoryTheory.Functo …
    -/
    dsimp
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₁.left j₁.hom
      e_1✝ : Eq (Quiver.Hom (F.obj j₁.left) d) (Quiver.Hom (F.obj j₁.left) ((Categor …
      ⊢ Eq j₁.hom (CategoryTheory.CategoryStruct.comp j₁.hom (CategoryTheory.Categor …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      ⊢ {j₁ j₂ : CategoryTheory.CostructuredArrow F d} → Quiver.Hom j₁ j₂ → (fun Y = …
    -/
  · intro j₁ j₂ f a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₂.left j₂.hom
      ⊢ Z j₁.left j₁.hom
    -/
    fapply h₂ _ _ _ _ f.left _ a
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₂.left j₂.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.left) j₂.hom) j₁.hom
    -/
    convert f.w
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₂.left j₂.hom
      e_1✝ : Eq (Quiver.Hom (F.obj j₁.left) d) (Quiver.Hom (F.obj j₁.left) ((Categor …
      ⊢ Eq j₁.hom (CategoryTheory.CategoryStruct.comp j₁.hom ((CategoryTheory.Functo …
    -/
    dsimp
    /-
      case h.e'_3.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      d : D
      Z : (X : C) → Quiver.Hom (F.obj X) d → Sort u_1
      h₁ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      h₂ : (X₁ X₂ : C) → (k₁ : Quiver.Hom (F.obj X₁) d) → (k₂ : Quiver.Hom (F.obj X₂ …
      X₀ : C
      k₀ : Quiver.Hom (F.obj X₀) d
      z : Z X₀ k₀
      j₁ j₂ : CategoryTheory.CostructuredArrow F d
      f : Quiver.Hom j₁ j₂
      a : Z j₂.left j₂.hom
      e_1✝ : Eq (Quiver.Hom (F.obj j₁.left) d) (Quiver.Hom (F.obj j₁.left) ((Categor …
      ⊢ Eq j₁.hom (CategoryTheory.CategoryStruct.comp j₁.hom (CategoryTheory.Categor …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given a cone over `F ⋙ G`, we can construct a `Cone G` with the same cocone point.
-/
@[simps]
def extendCone : Cone (F ⋙ G) ⥤ Cone G where
  obj c :=
    { pt := c.pt
      π :=
        { app := fun d => c.π.app (lift F d) ≫ G.map (homToLift F d)
          naturality := fun X Y f => by
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.Initial
              E : Type u₃
              inst✝ : CategoryTheory.Category.{v₃, u₃} E
              G : CategoryTheory.Functor D E
              c : CategoryTheory.Limits.Cone (F.comp G)
              X Y : D
              f : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const D).ob …
            -/
            dsimp; simp only [Category.id_comp, Category.assoc]
            -- This would be true if we'd chosen `lift F Y` to be `lift F X`
            -- and `homToLift F Y` to be `homToLift F X ≫ f`.
            apply
              induction F fun Z k =>
                (c.π.app Z ≫ G.map k : c.pt ⟶ _) =
                  c.π.app (lift F X) ≫ G.map (homToLift F X) ≫ G.map f
              /-
                case h₁
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Initial
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj X₁) Y) (k₂ : Quiver.Hom (F.obj X₂) Y)  …
              -/
            · intro Z₁ Z₂ k₁ k₂ g a z
              rw [← a, Functor.map_comp, ← Functor.comp_map, ← Category.assoc, ← Category.assoc,
                c.w] at z
              /-
                case h₁
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Initial
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                Z₁ Z₂ : C
                k₁ : Quiver.Hom (F.obj Z₁) Y
                k₂ : Quiver.Hom (F.obj Z₂) Y
                g : Quiver.Hom Z₁ Z₂
                a : Eq (CategoryTheory.CategoryStruct.comp (F.map g) k₂) k₁
                z : Eq (CategoryTheory.CategoryStruct.comp (c.π.app Z₂) (G.map k₂)) (CategoryT …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app Z₂) (G.map k₂)) (CategoryThe …
              -/
              rw [z, Category.assoc]
              /-
                🎉 no goals
              -/
              /-
                case h₂
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Initial
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj X₁) Y) (k₂ : Quiver.Hom (F.obj X₂) Y)  …
              -/
            · intro Z₁ Z₂ k₁ k₂ g a z
              rw [← a, Functor.map_comp, ← Functor.comp_map, ← Category.assoc, ← Category.assoc,
                c.w, z, Category.assoc]
              /-
                case z
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                F : CategoryTheory.Functor C D
                inst✝¹ : F.Initial
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                G : CategoryTheory.Functor D E
                c : CategoryTheory.Limits.Cone (F.comp G)
                X Y : D
                f : Quiver.Hom X Y
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app ?X₀) (G.map ?k₀)) (CategoryT …
              -/
            · rw [← Functor.map_comp] } }
              /-
                🎉 no goals
              -/
  map f := { hom := f.hom }


/-- Alternative equational lemma for `(extendCone c).π.app` in case a lift of the object
is given explicitly. -/
lemma extendCone_obj_π_app' (c : Cone (F ⋙ G)) {X : C} {Y : D} (f : F.obj X ⟶ Y) :
    (extendCone.obj c).π.app Y = c.π.app X ≫ G.map f := by
  apply induction (k₀ := f) (z := rfl) F fun Z g =>
    c.π.app Z ≫ G.map g = c.π.app X ≫ G.map f
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cone (F.comp G)
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj X₁) Y) (k₂ : Quiver.Hom (F.obj X₂) Y)  …
    -/
  · intro _ _ _ _ _ h₁ h₂
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cone (F.comp G)
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      X₁✝ X₂✝ : C
      k₁✝ : Quiver.Hom (F.obj X₁✝) Y
      k₂✝ : Quiver.Hom (F.obj X₂✝) Y
      f✝ : Quiver.Hom X₁✝ X₂✝
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map f✝) k₂✝) k₁✝
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (c.π.app X₁✝) (G.map k₁✝)) (Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app X₂✝) (G.map k₂✝)) (CategoryT …
    -/
    simp [← h₂, ← h₁, ← Functor.comp_map, c.π.naturality]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cone (F.comp G)
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj X₁) Y) (k₂ : Quiver.Hom (F.obj X₂) Y)  …
    -/
  · intro _ _ _ _ _ h₁ h₂
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      c : CategoryTheory.Limits.Cone (F.comp G)
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      X₁✝ X₂✝ : C
      k₁✝ : Quiver.Hom (F.obj X₁✝) Y
      k₂✝ : Quiver.Hom (F.obj X₂✝) Y
      f✝ : Quiver.Hom X₁✝ X₂✝
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map f✝) k₂✝) k₁✝
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (c.π.app X₂✝) (G.map k₂✝)) (Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app X₁✝) (G.map k₁✝)) (CategoryT …
    -/
    simp [← h₁, ← Functor.comp_map, c.π.naturality, h₂]
    /-
      🎉 no goals
    -/


@[simp]
theorem limit_cone_comp_aux (s : Cone (F ⋙ G)) (j : C) :
    s.π.app (lift F (F.obj j)) ≫ G.map (homToLift F (F.obj j)) = s.π.app j := by
  -- This point is that this would be true if we took `lift (F.obj j)` to just be `j`
  -- and `homToLift (F.obj j)` to be `𝟙 (F.obj j)`.
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Initial
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    s : CategoryTheory.Limits.Cone (F.comp G)
    j : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app (CategoryTheory.Functor.Init …
  -/
  apply induction F fun X k => s.π.app X ≫ G.map k = (s.π.app j : _)
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j : C
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj X₁) (F.obj j)) (k₂ : Quiver.Hom (F.obj …
    -/
  · intro j₁ j₂ k₁ k₂ f w h
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j₁) (F.obj j)
      k₂ : Quiver.Hom (F.obj j₂) (F.obj j)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) k₂) k₁
      h : Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₁) (G.map k₁)) (s.π.app j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₂) (G.map k₂)) (s.π.app j)
    -/
    rw [← s.w f]
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j₁) (F.obj j)
      k₂ : Quiver.Hom (F.obj j₂) (F.obj j)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) k₂) k₁
      h : Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₁) (G.map k₁)) (s.π.app j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← w] at h
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j₁) (F.obj j)
      k₂ : Quiver.Hom (F.obj j₂) (F.obj j)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) k₂) k₁
      h : Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₁) (G.map (CategoryTheory …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j : C
      ⊢ ∀ (X₁ X₂ : C) (k₁ : Quiver.Hom (F.obj X₁) (F.obj j)) (k₂ : Quiver.Hom (F.obj …
    -/
  · intro j₁ j₂ k₁ k₂ f w h
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j₁) (F.obj j)
      k₂ : Quiver.Hom (F.obj j₂) (F.obj j)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) k₂) k₁
      h : Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₂) (G.map k₂)) (s.π.app j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₁) (G.map k₁)) (s.π.app j)
    -/
    rw [← s.w f] at h
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j₁) (F.obj j)
      k₂ : Quiver.Hom (F.obj j₂) (F.obj j)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) k₂) k₁
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₁) (G.map k₁)) (s.π.app j)
    -/
    rw [← w]
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j j₁ j₂ : C
      k₁ : Quiver.Hom (F.obj j₁) (F.obj j)
      k₂ : Quiver.Hom (F.obj j₂) (F.obj j)
      f : Quiver.Hom j₁ j₂
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) k₂) k₁
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j₁) (G.map (CategoryTheory.C …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case z
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Initial
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      s : CategoryTheory.Limits.Cone (F.comp G)
      j : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app ?X₀) (G.map ?k₀)) (s.π.app j)
    -/
  · exact s.w (𝟙 _)
    /-
      🎉 no goals
    -/


/-- If `F` is initial,
the category of cones on `F ⋙ G` is equivalent to the category of cones on `G`,
for any `G : D ⥤ E`.
-/
@[simps]
def conesEquiv : Cone (F ⋙ G) ≌ Cone G where
  functor := extendCone
  inverse := Cones.whiskering F
                                          /-
                                            C : Type u₁
                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                            D : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                            F : CategoryTheory.Functor C D
                                            inst✝¹ : F.Initial
                                            E : Type u₃
                                            inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                            G : CategoryTheory.Functor D E
                                            c : CategoryTheory.Limits.Cone (F.comp G)
                                            ⊢ ∀ (j : C), Eq (((CategoryTheory.Functor.id (CategoryTheory.Limits.Cone (F.co …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents fun c => Cones.ext (Iso.refl _)
             /-
               🎉 no goals
             -/
                                            /-
                                              C : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                              D : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                              F : CategoryTheory.Functor C D
                                              inst✝¹ : F.Initial
                                              E : Type u₃
                                              inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                              G : CategoryTheory.Functor D E
                                              c : CategoryTheory.Limits.Cone G
                                              ⊢ ∀ (j : D), Eq ((((CategoryTheory.Limits.Cones.whiskering F).comp CategoryThe …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents fun c => Cones.ext (Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- When `F : C ⥤ D` is initial, and `t : Cone G` for some `G : D ⥤ E`,
`t.whisker F` is a limit cone exactly when `t` is.
-/
def isLimitWhiskerEquiv (t : Cone G) : IsLimit (t.whisker F) ≃ IsLimit t :=
  IsLimit.ofConeEquiv (conesEquiv F G).symm


/-- When `F` is initial, and `t : Cone (F ⋙ G)`,
`extendCone.obj t` is a limit cone exactly when `t` is.
-/
def isLimitExtendConeEquiv (t : Cone (F ⋙ G)) : IsLimit (extendCone.obj t) ≃ IsLimit t :=
  IsLimit.ofConeEquiv (conesEquiv F G)


/-- Given a limit cone over `G : D ⥤ E` we can construct a limit cone over `F ⋙ G`. -/
@[simps]
def limitConeComp (t : LimitCone G) : LimitCone (F ⋙ G) where
  cone := _
  isLimit := (isLimitWhiskerEquiv F _).symm t.isLimit


instance (priority := 100) comp_hasLimit [HasLimit G] : HasLimit (F ⋙ G) :=
  HasLimit.mk (limitConeComp F (getLimitCone G))


instance (priority := 100) comp_preservesLimit {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [PreservesLimit G H] : PreservesLimit (F ⋙ G) H where
  preserves {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesLimit G H
      c : CategoryTheory.Limits.Cone (F.comp G)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (H.mapCone c))
    -/
    refine ⟨isLimitExtendConeEquiv (G := G ⋙ H) F (H.mapCone c) ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesLimit G H
      c : CategoryTheory.Limits.Cone (F.comp G)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.Initial.extendCone.obj …
    -/
    let hc' := isLimitOfPreserves H ((isLimitExtendConeEquiv F c).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesLimit G H
      c : CategoryTheory.Limits.Cone (F.comp G)
      hc : CategoryTheory.Limits.IsLimit c
      hc' : CategoryTheory.Limits.IsLimit (H.mapCone (CategoryTheory.Functor.Initial …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.Initial.extendCone.obj …
    -/
    exact IsLimit.ofIsoLimit hc' (Cones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


instance (priority := 100) comp_reflectsLimit {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [ReflectsLimit G H] : ReflectsLimit (F ⋙ G) H where
  reflects {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsLimit G H
      c : CategoryTheory.Limits.Cone (F.comp G)
      hc : CategoryTheory.Limits.IsLimit (H.mapCone c)
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit c)
    -/
    refine ⟨isLimitExtendConeEquiv F _ (isLimitOfReflects H ?_)⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsLimit G H
      c : CategoryTheory.Limits.Cone (F.comp G)
      hc : CategoryTheory.Limits.IsLimit (H.mapCone c)
      ⊢ CategoryTheory.Limits.IsLimit (H.mapCone (CategoryTheory.Functor.Initial.ext …
    -/
    let hc' := (isLimitExtendConeEquiv (G := G ⋙ H) F _).symm hc
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsLimit G H
      c : CategoryTheory.Limits.Cone (F.comp G)
      hc : CategoryTheory.Limits.IsLimit (H.mapCone c)
      hc' : CategoryTheory.Limits.IsLimit (CategoryTheory.Functor.Initial.extendCone …
      ⊢ CategoryTheory.Limits.IsLimit (H.mapCone (CategoryTheory.Functor.Initial.ext …
    -/
    exact IsLimit.ofIsoLimit hc' (Cones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


instance (priority := 100) compCreatesLimit {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [CreatesLimit G H] : CreatesLimit (F ⋙ G) H where
  lifts {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit G H
      c : CategoryTheory.Limits.Cone ((F.comp G).comp H)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.LiftableCone (F.comp G) H c
    -/
    refine ⟨(liftLimit ((isLimitExtendConeEquiv F (G := G ⋙ H) _).symm hc)).whisker F, ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit G H
      c : CategoryTheory.Limits.Cone ((F.comp G).comp H)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Iso (H.mapCone (CategoryTheory.Limits.Cone.whisker F (Categor …
    -/
    let i := liftedLimitMapsToOriginal ((isLimitExtendConeEquiv F (G := G ⋙ H) _).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit G H
      c : CategoryTheory.Limits.Cone ((F.comp G).comp H)
      hc : CategoryTheory.Limits.IsLimit c
      i : CategoryTheory.Iso (H.mapCone (CategoryTheory.liftLimit ((CategoryTheory.F …
      ⊢ CategoryTheory.Iso (H.mapCone (CategoryTheory.Limits.Cone.whisker F (Categor …
    -/
    exact (Cones.whiskering F).mapIso i ≪≫ ((conesEquiv F (G ⋙ H)).unitIso.app _).symm
    /-
      🎉 no goals
    -/


instance limit_pre_isIso [HasLimit G] : IsIso (limit.pre G F) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Initial
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasLimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.limit.pre G F)
  -/
  rw [limit.pre_eq (limitConeComp F (getLimitCone G)) (getLimitCone G)]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Initial
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasLimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  erw [IsLimit.lift_self]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Initial
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasLimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : F.Initial
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.HasLimit G
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- When `F : C ⥤ D` is initial, and `G : D ⥤ E` has a limit, then `F ⋙ G` has a limit also and
`limit (F ⋙ G) ≅ limit G`

https://stacks.math.columbia.edu/tag/04E7
-/
@[simps! (config := .lemmasOnly)]
def limitIso [HasLimit G] : limit (F ⋙ G) ≅ limit G :=
  (asIso (limit.pre G F)).symm


/-- A pointfree version of `limitIso`, stating that whiskering by `F` followed by taking the
limit is isomorpic to taking the limit on the codomain of `F`. -/
def limIso [HasLimitsOfShape D E] [HasLimitsOfShape C E] :
    (whiskeringLeft _ _ _).obj F ⋙ lim ≅ lim (J := D) (C := E) :=
  Iso.symm <| NatIso.ofComponents (fun G => (limitIso F G).symm) fun f => by
    simp only [comp_obj, whiskeringLeft_obj_obj, lim_obj, comp_map, whiskeringLeft_obj_map, lim_map,
      Iso.symm_hom, limitIso_inv]
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape D E
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape C E
      X✝ Y✝ : CategoryTheory.Functor D E
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap f) (Cat …
    -/
    ext
    /-
      case w
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape D E
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape C E
      X✝ Y✝ : CategoryTheory.Functor D E
      f : Quiver.Hom X✝ Y✝
      j✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given a limit cone over `F ⋙ G` we can construct a limit cone over `G`. -/
@[simps]
def limitConeOfComp (t : LimitCone (F ⋙ G)) : LimitCone G where
  cone := extendCone.obj t.cone
  isLimit := (isLimitExtendConeEquiv F _).symm t.isLimit


/-- When `F` is initial, and `F ⋙ G` has a limit, then `G` has a limit also.

We can't make this an instance, because `F` is not determined by the goal.
(Even if this weren't a problem, it would cause a loop with `comp_hasLimit`.)
-/
theorem hasLimit_of_comp [HasLimit (F ⋙ G)] : HasLimit G :=
  HasLimit.mk (limitConeOfComp F (getLimitCone (F ⋙ G)))


theorem preservesLimit_of_comp {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [PreservesLimit (F ⋙ G) H] : PreservesLimit G H where
  preserves {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (H.mapCone c))
    -/
    refine ⟨isLimitWhiskerEquiv F _ ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker F (H.mapCo …
    -/
    let hc' := isLimitOfPreserves H ((isLimitWhiskerEquiv F _).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      hc' : CategoryTheory.Limits.IsLimit (H.mapCone (CategoryTheory.Limits.Cone.whi …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker F (H.mapCo …
    -/
    exact IsLimit.ofIsoLimit hc' (Cones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


theorem reflectsLimit_of_comp {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [ReflectsLimit (F ⋙ G) H] : ReflectsLimit G H where
  reflects {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit (H.mapCone c)
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit c)
    -/
    refine ⟨isLimitWhiskerEquiv F _ (isLimitOfReflects H ?_)⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit (H.mapCone c)
      ⊢ CategoryTheory.Limits.IsLimit (H.mapCone (CategoryTheory.Limits.Cone.whisker …
    -/
    let hc' := (isLimitWhiskerEquiv F _).symm hc
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.Limits.ReflectsLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit (H.mapCone c)
      hc' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker F (H.m …
      ⊢ CategoryTheory.Limits.IsLimit (H.mapCone (CategoryTheory.Limits.Cone.whisker …
    -/
    exact IsLimit.ofIsoLimit hc' (Cones.ext (Iso.refl _) (by simp))
    /-
      🎉 no goals
    -/


/-- If `F` is initial and `F ⋙ G` creates limits of `H`, then so does `G`. -/
def createsLimitOfComp {B : Type u₄} [Category.{v₄} B] {H : E ⥤ B}
    [CreatesLimit (F ⋙ G) H] : CreatesLimit G H where
  reflects := (reflectsLimit_of_comp F).reflects
  lifts {c} hc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone (G.comp H)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.LiftableCone G H c
    -/
    refine ⟨(extendCone (F := F)).obj (liftLimit ((isLimitWhiskerEquiv F _).symm hc)), ?_⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone (G.comp H)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Iso (H.mapCone (CategoryTheory.Functor.Initial.extendCone.obj …
    -/
    let i := liftedLimitMapsToOriginal (K := (F ⋙ G)) ((isLimitWhiskerEquiv F _).symm hc)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone (G.comp H)
      hc : CategoryTheory.Limits.IsLimit c
      i : CategoryTheory.Iso (H.mapCone (CategoryTheory.liftLimit ((CategoryTheory.F …
      ⊢ CategoryTheory.Iso (H.mapCone (CategoryTheory.Functor.Initial.extendCone.obj …
    -/
    refine ?_ ≪≫ ((extendCone (F := F)).mapIso i) ≪≫ ((conesEquiv F (G ⋙ H)).counitIso.app _)
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.Initial
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      G : CategoryTheory.Functor D E
      B : Type u₄
      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
      H : CategoryTheory.Functor E B
      inst✝ : CategoryTheory.CreatesLimit (F.comp G) H
      c : CategoryTheory.Limits.Cone (G.comp H)
      hc : CategoryTheory.Limits.IsLimit c
      i : CategoryTheory.Iso (H.mapCone (CategoryTheory.liftLimit ((CategoryTheory.F …
      ⊢ CategoryTheory.Iso (H.mapCone (CategoryTheory.Functor.Initial.extendCone.obj …
    -/
    exact Cones.ext (Iso.refl _)
    /-
      🎉 no goals
    -/


include F in
theorem hasLimitsOfShape_of_initial [HasLimitsOfShape C E] : HasLimitsOfShape D E where
  has_limit := fun _ => hasLimit_of_comp F


include F in
theorem preservesLimitsOfShape_of_initial {B : Type u₄} [Category.{v₄} B] (H : E ⥤ B)
    [PreservesLimitsOfShape C H] : PreservesLimitsOfShape D H where
  preservesLimit := preservesLimit_of_comp F


include F in
theorem reflectsLimitsOfShape_of_initial {B : Type u₄} [Category.{v₄} B] (H : E ⥤ B)
    [ReflectsLimitsOfShape C H] : ReflectsLimitsOfShape D H where
  reflectsLimit := reflectsLimit_of_comp F


include F in
/-- If `H` creates limits of shape `C` and `F : C ⥤ D` is initial, then `H` creates limits of shape
`D`. -/
def createsLimitsOfShapeOfInitial {B : Type u₄} [Category.{v₄} B] (H : E ⥤ B)
    [CreatesLimitsOfShape C H] : CreatesLimitsOfShape D H where
  CreatesLimit := createsLimitOfComp F


/-- The hypotheses also imply that `G` is final, see `final_of_comp_full_faithful'`. -/
theorem final_of_comp_full_faithful [Full G] [Faithful G] [Final (F ⋙ G)] : Final F where
  out d := isConnected_of_equivalent (StructuredArrow.post d F G).asEquivalence.symm


/-- The hypotheses also imply that `G` is initial, see `initial_of_comp_full_faithful'`. -/
theorem initial_of_comp_full_faithful [Full G] [Faithful G] [Initial (F ⋙ G)] : Initial F where
  out d := isConnected_of_equivalent (CostructuredArrow.post F G d).asEquivalence.symm


/-- See also the strictly more general `final_comp` below. -/
theorem final_comp_equivalence [Final F] [IsEquivalence G] : Final (F ⋙ G) :=
  let i : F ≅ (F ⋙ G) ⋙ G.inv := isoWhiskerLeft F G.asEquivalence.unitIso
  have : Final ((F ⋙ G) ⋙ G.inv) := final_of_natIso i
  final_of_comp_full_faithful (F ⋙ G) G.inv


/-- See also the strictly more general `initial_comp` below. -/
theorem initial_comp_equivalence [Initial F] [IsEquivalence G] : Initial (F ⋙ G) :=
  let i : F ≅ (F ⋙ G) ⋙ G.inv := isoWhiskerLeft F G.asEquivalence.unitIso
  have : Initial ((F ⋙ G) ⋙ G.inv) := initial_of_natIso i
  initial_of_comp_full_faithful (F ⋙ G) G.inv


/-- See also the strictly more general `final_comp` below. -/
theorem final_equivalence_comp [IsEquivalence F] [Final G] : Final (F ⋙ G) where
  out d := isConnected_of_equivalent (StructuredArrow.pre d F G).asEquivalence.symm


/-- See also the strictly more general `initial_comp` below. -/
theorem initial_equivalence_comp [IsEquivalence F] [Initial G] : Initial (F ⋙ G) where
  out d := isConnected_of_equivalent (CostructuredArrow.pre F G d).asEquivalence.symm


/-- See also the strictly more general `final_of_final_comp` below. -/
theorem final_of_equivalence_comp [IsEquivalence F] [Final (F ⋙ G)] : Final G where
  out d := isConnected_of_equivalent (StructuredArrow.pre d F G).asEquivalence


/-- See also the strictly more general `initial_of_initial_comp` below. -/
theorem initial_of_equivalence_comp [IsEquivalence F] [Initial (F ⋙ G)] : Initial G where
  out d := isConnected_of_equivalent (CostructuredArrow.pre F G d).asEquivalence


/-- See also the strictly more general `final_iff_comp_final_full_faithful` below. -/
theorem final_iff_comp_equivalence [IsEquivalence G] : Final F ↔ Final (F ⋙ G) :=
  ⟨fun _ => final_comp_equivalence _ _, fun _ => final_of_comp_full_faithful _ G⟩


/-- See also the strictly more general `final_iff_final_comp` below. -/
theorem final_iff_equivalence_comp [IsEquivalence F] : Final G ↔ Final (F ⋙ G) :=
  ⟨fun _ => final_equivalence_comp _ _, fun _ => final_of_equivalence_comp F _⟩


/-- See also the strictly more general `initial_iff_comp_initial_full_faithful` below. -/
theorem initial_iff_comp_equivalence [IsEquivalence G] : Initial F ↔ Initial (F ⋙ G) :=
  ⟨fun _ => initial_comp_equivalence _ _, fun _ => initial_of_comp_full_faithful _ G⟩


/-- See also the strictly more general `initial_iff_initial_comp` below. -/
theorem initial_iff_equivalence_comp [IsEquivalence F] : Initial G ↔ Initial (F ⋙ G) :=
  ⟨fun _ => initial_equivalence_comp _ _, fun _ => initial_of_equivalence_comp F _⟩


instance final_comp [hF : Final F] [hG : Final G] : Final (F ⋙ G) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    hF : F.Final
    hG : G.Final
    ⊢ (F.comp G).Final
  -/
  let s₁ : C ≌ AsSmall.{max u₁ v₁ u₂ v₂ u₃ v₃} C := AsSmall.equiv
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    hF : F.Final
    hG : G.Final
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    ⊢ (F.comp G).Final
  -/
  let s₂ : D ≌ AsSmall.{max u₁ v₁ u₂ v₂ u₃ v₃} D := AsSmall.equiv
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    hF : F.Final
    hG : G.Final
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    ⊢ (F.comp G).Final
  -/
  let s₃ : E ≌ AsSmall.{max u₁ v₁ u₂ v₂ u₃ v₃} E := AsSmall.equiv
  let i : s₁.inverse ⋙ (F ⋙ G) ⋙ s₃.functor ≅
      (s₁.inverse ⋙ F ⋙ s₂.functor) ⋙ (s₂.inverse ⋙ G ⋙ s₃.functor) :=
    isoWhiskerLeft (s₁.inverse ⋙ F) (isoWhiskerRight s₂.unitIso (G ⋙ s₃.functor))
  rw [final_iff_comp_equivalence (F ⋙ G) s₃.functor, final_iff_equivalence_comp s₁.inverse,
    final_natIso_iff i, final_iff_isIso_colimit_pre]
  rw [final_iff_comp_equivalence F s₂.functor, final_iff_equivalence_comp s₁.inverse,
    final_iff_isIso_colimit_pre] at hF
  rw [final_iff_comp_equivalence G s₃.functor, final_iff_equivalence_comp s₂.inverse,
    final_iff_isIso_colimit_pre] at hG
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    hG : ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (ma …
    i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.inv …
    ⊢ ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max ( …
  -/
  intro H
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    hG : ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (ma …
    i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.inv …
    H : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max (max (ma …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre H ((s₁.inverse.comp  …
  -/
  rw [← colimit.pre_pre]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    hG : ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (ma …
    i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.inv …
    H : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max (max (ma …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance initial_comp [Initial F] [Initial G] : Initial (F ⋙ G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Initial
    inst✝ : G.Initial
    ⊢ (F.comp G).Initial
  -/
  suffices Final (F ⋙ G).op from initial_of_final_op _
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Initial
    inst✝ : G.Initial
    ⊢ (F.comp G).op.Final
  -/
  exact final_comp F.op G.op
  /-
    🎉 no goals
  -/


theorem final_of_final_comp [hF : Final F] [hFG : Final (F ⋙ G)] : Final G := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    hF : F.Final
    hFG : (F.comp G).Final
    ⊢ G.Final
  -/
  let s₁ : C ≌ AsSmall.{max u₁ v₁ u₂ v₂ u₃ v₃} C := AsSmall.equiv
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    hF : F.Final
    hFG : (F.comp G).Final
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    ⊢ G.Final
  -/
  let s₂ : D ≌ AsSmall.{max u₁ v₁ u₂ v₂ u₃ v₃} D := AsSmall.equiv
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    hF : F.Final
    hFG : (F.comp G).Final
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    ⊢ G.Final
  -/
  let s₃ : E ≌ AsSmall.{max u₁ v₁ u₂ v₂ u₃ v₃} E := AsSmall.equiv
  let _i : s₁.inverse ⋙ (F ⋙ G) ⋙ s₃.functor ≅
      (s₁.inverse ⋙ F ⋙ s₂.functor) ⋙ (s₂.inverse ⋙ G ⋙ s₃.functor) :=
    isoWhiskerLeft (s₁.inverse ⋙ F) (isoWhiskerRight s₂.unitIso (G ⋙ s₃.functor))
  rw [final_iff_comp_equivalence G s₃.functor, final_iff_equivalence_comp s₂.inverse,
    final_iff_isIso_colimit_pre]
  rw [final_iff_comp_equivalence F s₂.functor, final_iff_equivalence_comp s₁.inverse,
    final_iff_isIso_colimit_pre] at hF
  rw [final_iff_comp_equivalence (F ⋙ G) s₃.functor, final_iff_equivalence_comp s₁.inverse,
    final_natIso_iff _i, final_iff_isIso_colimit_pre] at hFG
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    hFG : ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (m …
    _i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.in …
    ⊢ ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max ( …
  -/
  intro H
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    hFG : ∀ (G_1 : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (m …
    _i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.in …
    H : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max (max (ma …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre H (s₂.inverse.comp ( …
  -/
  replace hFG := hFG H
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    _i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.in …
    H : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max (max (ma …
    hFG : CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre H ((s₁.inverse.c …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre H (s₂.inverse.comp ( …
  -/
  rw [← colimit.pre_pre] at hFG
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    s₁ : CategoryTheory.Equivalence C (CategoryTheory.AsSmall C) := CategoryTheory …
    s₂ : CategoryTheory.Equivalence D (CategoryTheory.AsSmall D) := CategoryTheory …
    hF : ∀ (G : CategoryTheory.Functor (CategoryTheory.AsSmall D) (Type (max (max  …
    s₃ : CategoryTheory.Equivalence E (CategoryTheory.AsSmall E) := CategoryTheory …
    _i : CategoryTheory.Iso (s₁.inverse.comp ((F.comp G).comp s₃.functor)) ((s₁.in …
    H : CategoryTheory.Functor (CategoryTheory.AsSmall E) (Type (max (max (max (ma …
    hFG : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre H (s₂.inverse.comp ( …
  -/
  exact IsIso.of_isIso_comp_left (colimit.pre _ (s₁.inverse ⋙ F ⋙ s₂.functor)) _
  /-
    🎉 no goals
  -/


theorem initial_of_initial_comp [Initial F] [Initial (F ⋙ G)] : Initial G := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Initial
    inst✝ : (F.comp G).Initial
    ⊢ G.Initial
  -/
  suffices Final G.op from initial_of_final_op _
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Initial
    inst✝ : (F.comp G).Initial
    ⊢ G.op.Final
  -/
  have : Final (F.op ⋙ G.op) := show Final (F ⋙ G).op from inferInstance
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : F.Initial
    inst✝ : (F.comp G).Initial
    this : (F.op.comp G.op).Final
    ⊢ G.op.Final
  -/
  exact final_of_final_comp F.op G.op
  /-
    🎉 no goals
  -/


/-- The hypotheses also imply that `F` is final, see `final_of_comp_full_faithful`. -/
theorem final_of_comp_full_faithful' [Full G] [Faithful G] [Final (F ⋙ G)] : Final G :=
  have := final_of_comp_full_faithful F G
  final_of_final_comp F G


/-- The hypotheses also imply that `F` is initial, see `initial_of_comp_full_faithful`. -/
theorem initial_of_comp_full_faithful' [Full G] [Faithful G] [Initial (F ⋙ G)] : Initial G :=
  have := initial_of_comp_full_faithful F G
  initial_of_initial_comp F G


theorem final_iff_comp_final_full_faithful [Final G] [Full G] [Faithful G] :
    Final F ↔ Final (F ⋙ G) :=
  ⟨fun _ => final_comp _ _, fun _ => final_of_comp_full_faithful F G⟩


theorem initial_iff_comp_initial_full_faithful [Initial G] [Full G] [Faithful G] :
    Initial F ↔ Initial (F ⋙ G) :=
  ⟨fun _ => initial_comp _ _, fun _ => initial_of_comp_full_faithful F G⟩


theorem final_iff_final_comp [Final F] : Final G ↔ Final (F ⋙ G) :=
  ⟨fun _ => final_comp _ _, fun _ => final_of_final_comp F G⟩


theorem initial_iff_initial_comp [Initial F] : Initial G ↔ Initial (F ⋙ G) :=
  ⟨fun _ => initial_comp _ _, fun _ => initial_of_initial_comp F G⟩


/-- Final functors preserve filteredness.

This can be seen as a generalization of `IsFiltered.of_right_adjoint` (which states that right
adjoints preserve filteredness), as right adjoints are always final, see `final_of_adjunction`.
-/
theorem IsFilteredOrEmpty.of_final (F : C ⥤ D) [Final F] [IsFilteredOrEmpty C] :
    IsFilteredOrEmpty D where
  cocone_objs X Y := ⟨F.obj (IsFiltered.max (Final.lift F X) (Final.lift F Y)),
    Final.homToLift F X ≫ F.map (IsFiltered.leftToMax _ _),
    ⟨Final.homToLift F Y ≫ F.map (IsFiltered.rightToMax _ _), trivial⟩⟩
  cocone_maps {X Y} f g := by
    let P : StructuredArrow X F → Prop := fun h => ∃ (Z : C) (q₁ : h.right ⟶ Z)
      (q₂ : Final.lift F Y ⟶ Z), h.hom ≫ F.map q₁ = f ≫ Final.homToLift F Y ≫ F.map q₂
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      X Y : D
      f g : Quiver.Hom X Y
      P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
      ⊢ Exists fun Z => Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h)  …
    -/
    rsuffices ⟨Z, q₁, q₂, h⟩ : Nonempty (P (StructuredArrow.mk (g ≫ Final.homToLift F Y)))
    · refine ⟨F.obj (IsFiltered.coeq q₁ q₂),
        Final.homToLift F Y ≫ F.map (q₁ ≫ IsFiltered.coeqHom q₁ q₂), ?_⟩
      /-
        case intro.intro.intro.intro
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        Z : C
        q₁ : Quiver.Hom (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStr …
        q₂ : Quiver.Hom (CategoryTheory.Functor.Final.lift F Y) Z
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.mk  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      conv_lhs => rw [IsFiltered.coeq_condition]
      /-
        case intro.intro.intro.intro
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        Z : C
        q₁ : Quiver.Hom (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStr …
        q₂ : Quiver.Hom (CategoryTheory.Functor.Final.lift F Y) Z
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.mk  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      simp only [F.map_comp, ← reassoc_of% h, StructuredArrow.mk_hom_eq_self, Category.assoc]
      /-
        🎉 no goals
      -/
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      X Y : D
      f g : Quiver.Hom X Y
      P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
      ⊢ Nonempty (P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruc …
    -/
    have h₀ : P (StructuredArrow.mk (f ≫ Final.homToLift F Y)) := ⟨_, 𝟙 _, 𝟙 _, by simp⟩
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Final
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      X Y : D
      f g : Quiver.Hom X Y
      P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
      h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
      ⊢ Nonempty (P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruc …
    -/
    refine isPreconnected_induction P ?_ ?_ h₀ _
      /-
        case refine_1
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
        ⊢ ∀ {j₁ j₂ : CategoryTheory.StructuredArrow X F}, Quiver.Hom j₁ j₂ → P j₁ → P j₂
      -/
    · rintro U V h ⟨Z, q₁, q₂, hq⟩
      /-
        case refine_1.intro.intro.intro
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
        U V : CategoryTheory.StructuredArrow X F
        h : Quiver.Hom U V
        Z : C
        q₁ : Quiver.Hom U.right Z
        q₂ : Quiver.Hom (CategoryTheory.Functor.Final.lift F Y) Z
        hq : Eq (CategoryTheory.CategoryStruct.comp U.hom (F.map q₁)) (CategoryTheory. …
        ⊢ P V
      -/
      obtain ⟨W, q₃, q₄, hq'⟩ := IsFiltered.span q₁ h.right
      /-
        case refine_1.intro.intro.intro.intro.intro.intro
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
        U V : CategoryTheory.StructuredArrow X F
        h : Quiver.Hom U V
        Z : C
        q₁ : Quiver.Hom U.right Z
        q₂ : Quiver.Hom (CategoryTheory.Functor.Final.lift F Y) Z
        hq : Eq (CategoryTheory.CategoryStruct.comp U.hom (F.map q₁)) (CategoryTheory. …
        W : C
        q₃ : Quiver.Hom Z W
        q₄ : Quiver.Hom V.right W
        hq' : Eq (CategoryTheory.CategoryStruct.comp q₁ q₃) (CategoryTheory.CategorySt …
        ⊢ P V
      -/
      refine ⟨W, q₄, q₂ ≫ q₃, ?_⟩
      /-
        case refine_1.intro.intro.intro.intro.intro.intro
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
        U V : CategoryTheory.StructuredArrow X F
        h : Quiver.Hom U V
        Z : C
        q₁ : Quiver.Hom U.right Z
        q₂ : Quiver.Hom (CategoryTheory.Functor.Final.lift F Y) Z
        hq : Eq (CategoryTheory.CategoryStruct.comp U.hom (F.map q₁)) (CategoryTheory. …
        W : C
        q₃ : Quiver.Hom Z W
        q₄ : Quiver.Hom V.right W
        hq' : Eq (CategoryTheory.CategoryStruct.comp q₁ q₃) (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp V.hom (F.map q₄)) (CategoryTheory.Cat …
      -/
      rw [F.map_comp, ← reassoc_of% hq, ← F.map_comp, hq', F.map_comp, StructuredArrow.w_assoc]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
        ⊢ ∀ {j₁ j₂ : CategoryTheory.StructuredArrow X F}, Quiver.Hom j₁ j₂ → P j₂ → P j₁
      -/
    · rintro U V h ⟨Z, q₁, q₂, hq⟩
      /-
        case refine_2.intro.intro.intro
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Final
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        X Y : D
        f g : Quiver.Hom X Y
        P : CategoryTheory.StructuredArrow X F → Prop := fun h => Exists fun Z => Exis …
        h₀ : P (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
        U V : CategoryTheory.StructuredArrow X F
        h : Quiver.Hom U V
        Z : C
        q₁ : Quiver.Hom V.right Z
        q₂ : Quiver.Hom (CategoryTheory.Functor.Final.lift F Y) Z
        hq : Eq (CategoryTheory.CategoryStruct.comp V.hom (F.map q₁)) (CategoryTheory. …
        ⊢ P U
      -/
      exact ⟨Z, h.right ≫ q₁, q₂, by simp only [F.map_comp, StructuredArrow.w_assoc, hq]⟩
      /-
        🎉 no goals
      -/


/-- Final functors preserve filteredness.

This can be seen as a generalization of `IsFiltered.of_right_adjoint` (which states that right
adjoints preserve filteredness), as right adjoints are always final, see `final_of_adjunction`.
-/
theorem IsFiltered.of_final (F : C ⥤ D) [Final F] [IsFiltered C] : IsFiltered D :=
{ IsFilteredOrEmpty.of_final F with
  nonempty := Nonempty.map F.obj IsFiltered.nonempty }


/-- Initial functors preserve cofilteredness.

This can be seen as a generalization of `IsCofiltered.of_left_adjoint` (which states that left
adjoints preserve cofilteredness), as right adjoints are always initial,
see `initial_of_adjunction`.
-/
theorem IsCofilteredOrEmpty.of_initial (F : C ⥤ D) [Initial F] [IsCofilteredOrEmpty C] :
    IsCofilteredOrEmpty D :=
  have : IsFilteredOrEmpty Dᵒᵖ := IsFilteredOrEmpty.of_final F.op
  isCofilteredOrEmpty_of_isFilteredOrEmpty_op _


/-- Initial functors preserve cofilteredness.

This can be seen as a generalization of `IsCofiltered.of_left_adjoint` (which states that left
adjoints preserve cofilteredness), as right adjoints are always initial,
see `initial_of_adjunction`.
-/
theorem IsCofiltered.of_initial (F : C ⥤ D) [Initial F] [IsCofiltered C] : IsCofiltered D :=
  have : IsFiltered Dᵒᵖ := IsFiltered.of_final F.op
  isCofiltered_of_isFiltered_op _


/-- The functor `StructuredArrow.pre X T S` is final if `T` is final. -/
instance StructuredArrow.final_pre (T : C ⥤ D) [Final T] (S : D ⥤ E) (X : E) :
    Final (pre X T S) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    T : CategoryTheory.Functor C D
    inst✝ : T.Final
    S : CategoryTheory.Functor D E
    X : E
    ⊢ (CategoryTheory.StructuredArrow.pre X T S).Final
  -/
  refine ⟨fun f => ?_⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    T : CategoryTheory.Functor C D
    inst✝ : T.Final
    S : CategoryTheory.Functor D E
    X : E
    f : CategoryTheory.StructuredArrow X S
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow f (CategoryTheory …
  -/
  rw [isConnected_iff_of_equivalence (StructuredArrow.preEquivalence T f)]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    T : CategoryTheory.Functor C D
    inst✝ : T.Final
    S : CategoryTheory.Functor D E
    X : E
    f : CategoryTheory.StructuredArrow X S
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow f.right T)
  -/
  exact Final.out f.right
  /-
    🎉 no goals
  -/


/-- The functor `CostructuredArrow.pre X T S` is initial if `T` is initial. -/
instance CostructuredArrow.initial_pre (T : C ⥤ D) [Initial T] (S : D ⥤ E) (X : E) :
    Initial (CostructuredArrow.pre T S X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    T : CategoryTheory.Functor C D
    inst✝ : T.Initial
    S : CategoryTheory.Functor D E
    X : E
    ⊢ (CategoryTheory.CostructuredArrow.pre T S X).Initial
  -/
  refine ⟨fun f => ?_⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    T : CategoryTheory.Functor C D
    inst✝ : T.Initial
    S : CategoryTheory.Functor D E
    X : E
    f : CategoryTheory.CostructuredArrow S X
    ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (CategoryTheory …
  -/
  rw [isConnected_iff_of_equivalence (CostructuredArrow.preEquivalence T f)]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    T : CategoryTheory.Functor C D
    inst✝ : T.Initial
    S : CategoryTheory.Functor D E
    X : E
    f : CategoryTheory.CostructuredArrow S X
    ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow T f.left)
  -/
  exact Initial.out f.left
  /-
    🎉 no goals
  -/


/-- A prefunctor mapping structured arrows on `G` to structured arrows on `pre F G` with their
action on fibers being the identity. -/
def Grothendieck.structuredArrowToStructuredArrowPre (d : D) (f : F.obj d) :
    StructuredArrow d G ⥤q StructuredArrow ⟨d, f⟩ (pre F G) where
  obj := fun X => StructuredArrow.mk (Y := ⟨X.right, (F.map X.hom).obj f⟩)
                             /-
                               C : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                               D : Type u₂
                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                               F : CategoryTheory.Functor D CategoryTheory.Cat
                               G : CategoryTheory.Functor C D
                               d : D
                               f : ↑(F.obj d)
                               X : CategoryTheory.StructuredArrow d G
                               ⊢ Quiver.Hom { base := d, fiber := f }.base ((CategoryTheory.Grothendieck.pre  …
                             -/
                             /-
                               🎉 no goals
                             -/
    (Grothendieck.Hom.mk (by exact X.hom) (by dsimp; exact 𝟙 _))
                                                     /-
                                                       🎉 no goals
                                                     -/
  map := fun g => StructuredArrow.homMk
                             /-
                               C : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                               D : Type u₂
                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                               F : CategoryTheory.Functor D CategoryTheory.Cat
                               G : CategoryTheory.Functor C D
                               d : D
                               f : ↑(F.obj d)
                               X✝ Y✝ : CategoryTheory.StructuredArrow d G
                               g : Quiver.Hom X✝ Y✝
                               ⊢ Quiver.Hom ((fun X => CategoryTheory.StructuredArrow.mk { base := X.hom, fib …
                             -/
    (Grothendieck.Hom.mk (by exact g.right)
                             /-
                               🎉 no goals
                             -/
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F : CategoryTheory.Functor D CategoryTheory.Cat
                     G : CategoryTheory.Functor C D
                     d : D
                     f : ↑(F.obj d)
                     X✝ Y✝ : CategoryTheory.StructuredArrow d G
                     g : Quiver.Hom X✝ Y✝
                     ⊢ Eq (((G.comp F).map g.right).obj ((fun X => CategoryTheory.StructuredArrow.m …
                   -/
      (eqToHom (by dsimp; rw [← StructuredArrow.w g, map_comp, Cat.comp_obj])))
                          /-
                            🎉 no goals
                          -/
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor D CategoryTheory.Cat
          G : CategoryTheory.Functor C D
          d : D
          f : ↑(F.obj d)
          X✝ Y✝ : CategoryTheory.StructuredArrow d G
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => CategoryTheory.StructuredA …
        -/
    (by simp only [StructuredArrow.mk_right]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor D CategoryTheory.Cat
          G : CategoryTheory.Functor C D
          d : D
          f : ↑(F.obj d)
          X✝ Y✝ : CategoryTheory.StructuredArrow d G
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.mk {  …
        -/
                                   /-
                                     🎉 no goals
                                   -/
        apply Grothendieck.ext <;> simp)
                                   /-
                                     🎉 no goals
                                   -/


instance Grothendieck.final_pre [hG : Final G] : (Grothendieck.pre F G).Final := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D CategoryTheory.Cat
    G : CategoryTheory.Functor C D
    hG : G.Final
    ⊢ (CategoryTheory.Grothendieck.pre F G).Final
  -/
  constructor
  /-
    case out
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D CategoryTheory.Cat
    G : CategoryTheory.Functor C D
    hG : G.Final
    ⊢ ∀ (d : CategoryTheory.Grothendieck F), CategoryTheory.IsConnected (CategoryT …
  -/
  rintro ⟨d, f⟩
  /-
    case out.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D CategoryTheory.Cat
    G : CategoryTheory.Functor C D
    hG : G.Final
    d : D
    f : ↑(F.obj d)
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow { base := d, fibe …
  -/
  let ⟨u, c, g⟩ : Nonempty (StructuredArrow d G) := inferInstance
  letI :  Nonempty (StructuredArrow ⟨d, f⟩ (pre F G)) :=
    ⟨u, ⟨c, (F.map g).obj f⟩, ⟨(by exact g), (by exact 𝟙 _)⟩⟩
  /-
    case out.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D CategoryTheory.Cat
    G : CategoryTheory.Functor C D
    hG : G.Final
    d : D
    f : ↑(F.obj d)
    u : CategoryTheory.Discrete PUnit.{1}
    c : C
    g : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj u) (G.obj c)
    this : Nonempty (CategoryTheory.StructuredArrow { base := d, fiber := f } (Cat …
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow { base := d, fibe …
  -/
  apply zigzag_isConnected
  /-
    case out.mk.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D CategoryTheory.Cat
    G : CategoryTheory.Functor C D
    hG : G.Final
    d : D
    f : ↑(F.obj d)
    u : CategoryTheory.Discrete PUnit.{1}
    c : C
    g : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj u) (G.obj c)
    this : Nonempty (CategoryTheory.StructuredArrow { base := d, fiber := f } (Cat …
    ⊢ ∀ (j₁ j₂ : CategoryTheory.StructuredArrow { base := d, fiber := f } (Categor …
  -/
  rintro ⟨⟨⟨⟩⟩, ⟨bi, fi⟩, ⟨gbi, gfi⟩⟩ ⟨⟨⟨⟩⟩, ⟨bj, fj⟩, ⟨gbj, gfj⟩⟩
  /-
    case out.mk.h.mk.mk.unit.mk.mk.mk.mk.unit.mk.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D CategoryTheory.Cat
    G : CategoryTheory.Functor C D
    hG : G.Final
    d : D
    f : ↑(F.obj d)
    u : CategoryTheory.Discrete PUnit.{1}
    c : C
    g : Quiver.Hom ((CategoryTheory.Functor.fromPUnit d).obj u) (G.obj c)
    this : Nonempty (CategoryTheory.StructuredArrow { base := d, fiber := f } (Cat …
    bi : C
    fi : ↑((G.comp F).obj bi)
    gbi : Quiver.Hom ((CategoryTheory.Functor.fromPUnit { base := d, fiber := f }) …
    gfi : Quiver.Hom ((F.map gbi).obj ((CategoryTheory.Functor.fromPUnit { base := …
    bj : C
    fj : ↑((G.comp F).obj bj)
    gbj : Quiver.Hom ((CategoryTheory.Functor.fromPUnit { base := d, fiber := f }) …
    gfj : Quiver.Hom ((F.map gbj).obj ((CategoryTheory.Functor.fromPUnit { base := …
    ⊢ CategoryTheory.Zigzag { left := { as := PUnit.unit }, right := { base := bi, …
  -/
  dsimp at fj fi gfi gbi gbj gfj
  apply Zigzag.trans (j₂ := StructuredArrow.mk (Y := ⟨bi, ((F.map gbi).obj f)⟩)
      (Grothendieck.Hom.mk gbi (𝟙 _)))
    (.of_zag (.inr ⟨StructuredArrow.homMk (Grothendieck.Hom.mk (by dsimp; exact 𝟙 _)
      (eqToHom (by simp) ≫ gfi)) (by apply Grothendieck.ext <;> simp)⟩))
  refine Zigzag.trans (j₂ := StructuredArrow.mk (Y := ⟨bj, ((F.map gbj).obj f)⟩)
      (Grothendieck.Hom.mk gbj (𝟙 _))) ?_
    (.of_zag (.inl ⟨StructuredArrow.homMk (Grothendieck.Hom.mk (by dsimp; exact 𝟙 _)
      (eqToHom (by simp) ≫ gfj)) (by apply Grothendieck.ext <;> simp)⟩))
  exact zigzag_prefunctor_obj_of_zigzag (Grothendieck.structuredArrowToStructuredArrowPre F G d f)
    (isPreconnected_zigzag (.mk gbi) (.mk gbj))


