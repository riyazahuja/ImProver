instance [IsFilteredOrEmpty C] [∀ c, IsFilteredOrEmpty (F.obj c)] :
    IsFilteredOrEmpty (Grothendieck F) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
    inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
    ⊢ CategoryTheory.IsFilteredOrEmpty (CategoryTheory.Grothendieck F)
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
      inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
      ⊢ ∀ (X Y : CategoryTheory.Grothendieck F), Exists fun Z => Exists fun x => Exi …
    -/
  · rintro ⟨c, f⟩ ⟨d, g⟩
    exact ⟨⟨max c d, max ((F.map (leftToMax c d)).obj f) ((F.map (rightToMax c d)).obj g)⟩,
      ⟨leftToMax c d, leftToMax _ _⟩, ⟨rightToMax c d, rightToMax _ _⟩, trivial⟩
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
      inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
      ⊢ ∀ ⦃X Y : CategoryTheory.Grothendieck F⦄ (f g : Quiver.Hom X Y), Exists fun Z …
    -/
  · rintro ⟨c, f⟩ ⟨d, g⟩ ⟨u, x⟩ ⟨v, y⟩
    refine ⟨⟨coeq u v, coeq (eqToHom ?_ ≫
        (F.map (coeqHom u v)).map x) ((F.map (coeqHom u v)).map y)⟩, ⟨coeqHom u v, coeqHom _ _⟩, ?_⟩
      /-
        case refine_2.mk.mk.mk.mk.refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor C CategoryTheory.Cat
        inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
        inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
        c : C
        f : ↑(F.obj c)
        d : C
        g : ↑(F.obj d)
        u : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        x : Quiver.Hom ((F.map u).obj { base := c, fiber := f }.fiber) { base := d, fi …
        v : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        y : Quiver.Hom ((F.map v).obj { base := c, fiber := f }.fiber) { base := d, fi …
        ⊢ Eq ((F.map (CategoryTheory.IsFiltered.coeqHom u v)).obj ((F.map v).obj { bas …
      -/
    · conv_rhs => rw [← Cat.comp_obj, ← F.map_comp, coeq_condition, F.map_comp, Cat.comp_obj]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.mk.mk.mk.mk.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor C CategoryTheory.Cat
        inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
        inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
        c : C
        f : ↑(F.obj c)
        d : C
        g : ↑(F.obj d)
        u : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        x : Quiver.Hom ((F.map u).obj { base := c, fiber := f }.fiber) { base := d, fi …
        v : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        y : Quiver.Hom ((F.map v).obj { base := c, fiber := f }.fiber) { base := d, fi …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { base := u, fiber := x } { base := C …
      -/
    · apply Grothendieck.ext _ _ (coeq_condition u v)
      /-
        case refine_2.mk.mk.mk.mk.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor C CategoryTheory.Cat
        inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
        inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
        c : C
        f : ↑(F.obj c)
        d : C
        g : ↑(F.obj d)
        u : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        x : Quiver.Hom ((F.map u).obj { base := c, fiber := f }.fiber) { base := d, fi …
        v : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        y : Quiver.Hom ((F.map v).obj { base := c, fiber := f }.fiber) { base := d, fi …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      refine Eq.trans ?_ (eqToHom _ ≫= coeq_condition _ _)
      /-
        case refine_2.mk.mk.mk.mk.refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor C CategoryTheory.Cat
        inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
        inst✝ : ∀ (c : C), CategoryTheory.IsFilteredOrEmpty ↑(F.obj c)
        c : C
        f : ↑(F.obj c)
        d : C
        g : ↑(F.obj d)
        u : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        x : Quiver.Hom ((F.map u).obj { base := c, fiber := f }.fiber) { base := d, fi …
        v : Quiver.Hom { base := c, fiber := f }.base { base := d, fiber := g }.base
        y : Quiver.Hom ((F.map v).obj { base := c, fiber := f }.fiber) { base := d, fi …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp
      /-
        🎉 no goals
      -/


instance [IsFiltered C] [∀ c, IsFiltered (F.obj c)] : IsFiltered (Grothendieck F) := by
  have : Nonempty (Grothendieck F) := by
    obtain ⟨c⟩ : Nonempty C := IsFiltered.nonempty
    obtain ⟨f⟩ : Nonempty (F.obj c) := IsFiltered.nonempty
    exact ⟨⟨c, f⟩⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    inst✝¹ : CategoryTheory.IsFiltered C
    inst✝ : ∀ (c : C), CategoryTheory.IsFiltered ↑(F.obj c)
    this : Nonempty (CategoryTheory.Grothendieck F)
    ⊢ CategoryTheory.IsFiltered (CategoryTheory.Grothendieck F)
  -/
  apply IsFiltered.mk
  /-
    🎉 no goals
  -/


