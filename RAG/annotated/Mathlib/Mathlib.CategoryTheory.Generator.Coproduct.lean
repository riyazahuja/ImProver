open Classical in
lemma isSeparator_of_isColimit_cofan
    (hS : IsSeparating (Set.range S)) {c : Cofan S} (hc : IsColimit c) :
    IsSeparator c.pt := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ι : Type w
    S : ι → C
    hS : CategoryTheory.IsSeparating (Set.range S)
    c : CategoryTheory.Limits.Cofan S
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsSeparator c.pt
  -/
  intro X Y f g h
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ι : Type w
    S : ι → C
    hS : CategoryTheory.IsSeparating (Set.range S)
    c : CategoryTheory.Limits.Cofan S
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    f g : Quiver.Hom X Y
    h : ∀ (G : C), Membership.mem (Singleton.singleton c.pt) G → ∀ (h : Quiver.Hom …
    ⊢ Eq f g
  -/
  apply hS
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ι : Type w
    S : ι → C
    hS : CategoryTheory.IsSeparating (Set.range S)
    c : CategoryTheory.Limits.Cofan S
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    f g : Quiver.Hom X Y
    h : ∀ (G : C), Membership.mem (Singleton.singleton c.pt) G → ∀ (h : Quiver.Hom …
    ⊢ ∀ (G : C), Membership.mem (Set.range S) G → ∀ (h : Quiver.Hom G X), Eq (Cate …
  -/
  rintro _ ⟨i, rfl⟩ α
  let β : c.pt ⟶ X := Cofan.IsColimit.desc hc
      (fun j ↦ if hij : i = j then eqToHom (by rw [hij]) ≫ α else 0)
  /-
    case a.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ι : Type w
    S : ι → C
    hS : CategoryTheory.IsSeparating (Set.range S)
    c : CategoryTheory.Limits.Cofan S
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    f g : Quiver.Hom X Y
    h : ∀ (G : C), Membership.mem (Singleton.singleton c.pt) G → ∀ (h : Quiver.Hom …
    i : ι
    α : Quiver.Hom (S i) X
    β : Quiver.Hom c.pt X := CategoryTheory.Limits.Cofan.IsColimit.desc hc fun j = …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α f) (CategoryTheory.CategoryStruct.c …
  -/
  have hβ : c.inj i ≫ β = α := by simp [β]
  /-
    case a.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    ι : Type w
    S : ι → C
    hS : CategoryTheory.IsSeparating (Set.range S)
    c : CategoryTheory.Limits.Cofan S
    hc : CategoryTheory.Limits.IsColimit c
    X Y : C
    f g : Quiver.Hom X Y
    h : ∀ (G : C), Membership.mem (Singleton.singleton c.pt) G → ∀ (h : Quiver.Hom …
    i : ι
    α : Quiver.Hom (S i) X
    β : Quiver.Hom c.pt X := CategoryTheory.Limits.Cofan.IsColimit.desc hc fun j = …
    hβ : Eq (CategoryTheory.CategoryStruct.comp (c.inj i) β) α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α f) (CategoryTheory.CategoryStruct.c …
  -/
  simp only [← hβ, Category.assoc, h c.pt (by simp) β]
  /-
    🎉 no goals
  -/


lemma isSeparator_coproduct (hS : IsSeparating (Set.range S)) [HasCoproduct S] :
    IsSeparator (∐ S) :=
  isSeparator_of_isColimit_cofan hS (colimit.isColimit _)


