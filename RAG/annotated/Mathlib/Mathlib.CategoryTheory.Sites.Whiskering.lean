/-- Describes the property of a functor to "preserve sheaves". -/
class GrothendieckTopology.HasSheafCompose : Prop where
  /-- For every sheaf `P`, `P ⋙ F` is a sheaf. -/
  isSheaf (P : Cᵒᵖ ⥤ A) (hP : Presheaf.IsSheaf J P) : Presheaf.IsSheaf J (P ⋙ F)


/-- Composing a functor which `HasSheafCompose`, yields a functor between sheaf categories. -/
@[simps]
def sheafCompose : Sheaf J A ⥤ Sheaf J B where
  obj G := ⟨G.val ⋙ F, GrothendieckTopology.HasSheafCompose.isSheaf G.val G.2⟩
  map η := ⟨whiskerRight η.val _⟩
  map_id _ := Sheaf.Hom.ext <| whiskerRight_id _
  map_comp _ _ := Sheaf.Hom.ext <| whiskerRight_comp _ _ _


instance [F.Faithful] : (sheafCompose J F ⋙ sheafToPresheaf _ _).Faithful :=
  show (sheafToPresheaf _ _ ⋙ (whiskeringRight Cᵒᵖ A B).obj F).Faithful from inferInstance


instance [F.Faithful] [F.Full] : (sheafCompose J F ⋙ sheafToPresheaf _ _).Full :=
  show (sheafToPresheaf _ _ ⋙ (whiskeringRight Cᵒᵖ A B).obj F).Full from inferInstance


instance [F.Faithful] : (sheafCompose J F).Faithful :=
  Functor.Faithful.of_comp (sheafCompose J F) (sheafToPresheaf _ _)


instance [F.Full] [F.Faithful] : (sheafCompose J F).Full :=
  Functor.Full.of_comp_faithful (sheafCompose J F) (sheafToPresheaf _ _)


instance [F.ReflectsIsomorphisms] : (sheafCompose J F).ReflectsIsomorphisms where
  reflects {G₁ G₂} f _ := by
    rw [← isIso_iff_of_reflects_iso _ (sheafToPresheaf _ _),
      ← isIso_iff_of_reflects_iso _ ((whiskeringRight Cᵒᵖ A B).obj F)]
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F G H : CategoryTheory.Functor A B
      η : Quiver.Hom F G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      inst✝ : F.ReflectsIsomorphisms
      G₁ G₂ : CategoryTheory.Sheaf J A
      f : Quiver.Hom G₁ G₂
      x✝ : CategoryTheory.IsIso ((CategoryTheory.sheafCompose J F).map f)
      ⊢ CategoryTheory.IsIso (((CategoryTheory.whiskeringRight (Opposite C) A B).obj …
    -/
    change IsIso ((sheafToPresheaf _ _).map ((sheafCompose J F).map f))
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F G H : CategoryTheory.Functor A B
      η : Quiver.Hom F G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      inst✝ : F.ReflectsIsomorphisms
      G₁ G₂ : CategoryTheory.Sheaf J A
      f : Quiver.Hom G₁ G₂
      x✝ : CategoryTheory.IsIso ((CategoryTheory.sheafCompose J F).map f)
      ⊢ CategoryTheory.IsIso ((CategoryTheory.sheafToPresheaf J B).map ((CategoryThe …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/--
If `η : F ⟶ G` is a natural transformation then we obtain a morphism of functors
`sheafCompose J F ⟶ sheafCompose J G` by whiskering with `η` on the level of presheaves.
-/
def sheafCompose_map : sheafCompose J F ⟶ sheafCompose J G where
  app := fun _ => .mk <| whiskerLeft _ η


@[simp]
lemma sheafCompose_id : sheafCompose_map (F := F) J (𝟙 _) = 𝟙 _ := rfl


@[simp]
lemma sheafCompose_comp :
    sheafCompose_map J (η ≫ γ) = sheafCompose_map J η ≫ sheafCompose_map J γ := rfl


/-- The multicospan associated to a cover `S : J.Cover X` and a presheaf of the form `P ⋙ F`
is isomorphic to the composition of the multicospan associated to `S` and `P`,
composed with `F`. -/
@[simps!]
def multicospanComp : (S.index (P ⋙ F)).multicospan ≅ (S.index P).multicospan ⋙ F :=
  NatIso.ofComponents
    (fun t =>
      match t with
      | WalkingMulticospan.left _ => Iso.refl _
      | WalkingMulticospan.right _ => Iso.refl _)
    (by
      /-
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} A
        B : Type u₃
        inst✝³ : CategoryTheory.Category.{v₃, u₃} B
        J : CategoryTheory.GrothendieckTopology C
        U : C
        R : CategoryTheory.Presieve U
        F G H : CategoryTheory.Functor A B
        η : Quiver.Hom F G
        γ : Quiver.Hom G H
        inst✝² : J.HasSheafCompose F
        inst✝¹ : J.HasSheafCompose G
        inst✝ : J.HasSheafCompose H
        P : CategoryTheory.Functor (Opposite C) A
        X : C
        S : J.Cover X
        ⊢ ∀ {X_1 Y : CategoryTheory.Limits.WalkingMulticospan (S.index (P.comp F)).fst …
      -/
      rintro (a | b) (a | b) (f | f | f)
      /-
        case left.left.id
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        A : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} A
        B : Type u₃
        inst✝³ : CategoryTheory.Category.{v₃, u₃} B
        J : CategoryTheory.GrothendieckTopology C
        U : C
        R : CategoryTheory.Presieve U
        F G H : CategoryTheory.Functor A B
        η : Quiver.Hom F G
        γ : Quiver.Hom G H
        inst✝² : J.HasSheafCompose F
        inst✝¹ : J.HasSheafCompose G
        inst✝ : J.HasSheafCompose H
        P : CategoryTheory.Functor (Opposite C) A
        X : C
        S : J.Cover X
        a : (S.index (P.comp F)).L
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((S.index (P.comp F)).multicospan.map …
      -/
      all_goals aesop_cat)
      /-
        🎉 no goals
      -/


/-- Mapping the multifork associated to a cover `S : J.Cover X` and a presheaf `P` with
respect to a functor `F` is isomorphic (upto a natural isomorphism of the underlying functors)
to the multifork associated to `S` and `P ⋙ F`. -/
def mapMultifork :
    F.mapCone (S.multifork P) ≅
      (Limits.Cones.postcompose (S.multicospanComp F P).hom).obj (S.multifork (P ⋙ F)) :=
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} A
    B : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} B
    J : CategoryTheory.GrothendieckTopology C
    U : C
    R : CategoryTheory.Presieve U
    F G H : CategoryTheory.Functor A B
    η : Quiver.Hom F G
    γ : Quiver.Hom G H
    inst✝² : J.HasSheafCompose F
    inst✝¹ : J.HasSheafCompose G
    inst✝ : J.HasSheafCompose H
    P : CategoryTheory.Functor (Opposite C) A
    X : C
    S : J.Cover X
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingMulticospan (S.index P).fstTo (S.index P …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/--
Composing a sheaf with a functor preserving the limit of `(S.index P).multicospan` yields a functor
between sheaf categories.
-/
instance hasSheafCompose_of_preservesMulticospan (F : A ⥤ B)
    [∀ (X : C) (S : J.Cover X) (P : Cᵒᵖ ⥤ A), PreservesLimit (S.index P).multicospan F] :
    J.HasSheafCompose F where
  isSheaf P hP := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F✝ G H : CategoryTheory.Functor A B
      η : Quiver.Hom F✝ G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F✝
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      F : CategoryTheory.Functor A B
      inst✝ : ∀ (X : C) (S : J.Cover X) (P : CategoryTheory.Functor (Opposite C) A), …
      P : CategoryTheory.Functor (Opposite C) A
      hP : CategoryTheory.Presheaf.IsSheaf J P
      ⊢ CategoryTheory.Presheaf.IsSheaf J (P.comp F)
    -/
    rw [Presheaf.isSheaf_iff_multifork] at hP ⊢
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F✝ G H : CategoryTheory.Functor A B
      η : Quiver.Hom F✝ G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F✝
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      F : CategoryTheory.Functor A B
      inst✝ : ∀ (X : C) (S : J.Cover X) (P : CategoryTheory.Functor (Opposite C) A), …
      P : CategoryTheory.Functor (Opposite C) A
      hP : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mul …
      ⊢ ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.multif …
    -/
    intro X S
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F✝ G H : CategoryTheory.Functor A B
      η : Quiver.Hom F✝ G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F✝
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      F : CategoryTheory.Functor A B
      inst✝ : ∀ (X : C) (S : J.Cover X) (P : CategoryTheory.Functor (Opposite C) A), …
      P : CategoryTheory.Functor (Opposite C) A
      hP : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mul …
      X : C
      S : J.Cover X
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork (P.comp F)))
    -/
    obtain ⟨h⟩ := hP X S
    /-
      case intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F✝ G H : CategoryTheory.Functor A B
      η : Quiver.Hom F✝ G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F✝
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      F : CategoryTheory.Functor A B
      inst✝ : ∀ (X : C) (S : J.Cover X) (P : CategoryTheory.Functor (Opposite C) A), …
      P : CategoryTheory.Functor (Opposite C) A
      hP : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mul …
      X : C
      S : J.Cover X
      h : CategoryTheory.Limits.IsLimit (S.multifork P)
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork (P.comp F)))
    -/
    replace h := isLimitOfPreserves F h
    /-
      case intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F✝ G H : CategoryTheory.Functor A B
      η : Quiver.Hom F✝ G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F✝
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      F : CategoryTheory.Functor A B
      inst✝ : ∀ (X : C) (S : J.Cover X) (P : CategoryTheory.Functor (Opposite C) A), …
      P : CategoryTheory.Functor (Opposite C) A
      hP : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mul …
      X : C
      S : J.Cover X
      h : CategoryTheory.Limits.IsLimit (F.mapCone (S.multifork P))
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork (P.comp F)))
    -/
    replace h := Limits.IsLimit.ofIsoLimit h (S.mapMultifork F P)
    /-
      case intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} A
      B : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} B
      J : CategoryTheory.GrothendieckTopology C
      U : C
      R : CategoryTheory.Presieve U
      F✝ G H : CategoryTheory.Functor A B
      η : Quiver.Hom F✝ G
      γ : Quiver.Hom G H
      inst✝³ : J.HasSheafCompose F✝
      inst✝² : J.HasSheafCompose G
      inst✝¹ : J.HasSheafCompose H
      F : CategoryTheory.Functor A B
      inst✝ : ∀ (X : C) (S : J.Cover X) (P : CategoryTheory.Functor (Opposite C) A), …
      P : CategoryTheory.Functor (Opposite C) A
      hP : ∀ (X : C) (S : J.Cover X), Nonempty (CategoryTheory.Limits.IsLimit (S.mul …
      X : C
      S : J.Cover X
      h : CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (C …
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (S.multifork (P.comp F)))
    -/
    exact ⟨Limits.IsLimit.postcomposeHomEquiv (S.multicospanComp F P) _ h⟩
    /-
      🎉 no goals
    -/


/--
Composing a sheaf with a functor preserving limits of the same size as the hom sets in `C` yields a
functor between sheaf categories.

Note: the size of the limit that `F` is required to preserve in
`hasSheafCompose_of_preservesMulticospan` is in general larger than this.
-/
instance hasSheafCompose_of_preservesLimitsOfSize [PreservesLimitsOfSize.{v₁, max u₁ v₁} F] :
    J.HasSheafCompose F where
  isSheaf _ hP := Presheaf.isSheaf_comp_of_isSheaf J _ F hP


lemma Sheaf.isSeparated [ConcreteCategory A] [J.HasSheafCompose (forget A)]
    (F : Sheaf J A) : Presheaf.IsSeparated J F.val := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    inst✝¹ : CategoryTheory.ConcreteCategory A
    inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
    F : CategoryTheory.Sheaf J A
    ⊢ CategoryTheory.Presheaf.IsSeparated J F.val
  -/
  rintro X S hS x y h
  exact (Presieve.isSeparated_of_isSheaf _ _ ((isSheaf_iff_isSheaf_of_type _ _).1
    ((sheafCompose J (forget A)).obj F).2) S hS).ext (fun _ _ hf => h _ _ hf)


lemma Presheaf.IsSheaf.isSeparated {F : Cᵒᵖ ⥤ A} [ConcreteCategory A]
    [J.HasSheafCompose (forget A)] (hF : Presheaf.IsSheaf J F) :
    Presheaf.IsSeparated J F :=
  Sheaf.isSeparated ⟨F, hF⟩


