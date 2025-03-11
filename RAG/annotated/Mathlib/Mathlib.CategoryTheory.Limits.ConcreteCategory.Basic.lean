/-- If a functor `G : J ⥤ C` to a concrete category has a limit and that `forget C`
is corepresentable, then `(G ⋙ forget C).sections` is small. -/
lemma small_sections_of_hasLimit
    {C : Type u} [Category.{v} C] [ConcreteCategory.{v} C]
    [(forget C).IsCorepresentable] {J : Type w} [Category.{t} J] (G : J ⥤ C) [HasLimit G] :
    Small.{v} (G ⋙ forget C).sections := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : (CategoryTheory.forget C).IsCorepresentable
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    G : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit G
    ⊢ Small.{v, max v w} ↑(G.comp (CategoryTheory.forget C)).sections
  -/
  rw [← Types.hasLimit_iff_small_sections]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : (CategoryTheory.forget C).IsCorepresentable
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    G : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit G
    ⊢ CategoryTheory.Limits.HasLimit (G.comp (CategoryTheory.forget C))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem to_product_injective_of_isLimit {D : Cone F} (hD : IsLimit D) :
    Function.Injective fun (x : D.pt) (j : J) => D.π.app j x := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    ⊢ Function.Injective fun x j => (D.π.app j) x
  -/
  let E := (forget C).mapCone D
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    ⊢ Function.Injective fun x j => (D.π.app j) x
  -/
  let hE : IsLimit E := isLimitOfPreserves _ hD
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    hE : CategoryTheory.Limits.IsLimit E := CategoryTheory.Limits.isLimitOfPreserv …
    ⊢ Function.Injective fun x j => (D.π.app j) x
  -/
  let G := Types.limitCone.{w, v} (F ⋙ forget C)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    hE : CategoryTheory.Limits.IsLimit E := CategoryTheory.Limits.isLimitOfPreserv …
    G : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := CategoryT …
    ⊢ Function.Injective fun x j => (D.π.app j) x
  -/
  let hG := Types.limitConeIsLimit.{w, v} (F ⋙ forget C)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    hE : CategoryTheory.Limits.IsLimit E := CategoryTheory.Limits.isLimitOfPreserv …
    G : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := CategoryT …
    hG : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.limitCone (F.c …
    ⊢ Function.Injective fun x j => (D.π.app j) x
  -/
  let T : E.pt ≅ G.pt := hE.conePointUniqueUpToIso hG
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    hE : CategoryTheory.Limits.IsLimit E := CategoryTheory.Limits.isLimitOfPreserv …
    G : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := CategoryT …
    hG : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.limitCone (F.c …
    T : CategoryTheory.Iso E.pt G.pt := hE.conePointUniqueUpToIso hG
    ⊢ Function.Injective fun x j => (D.π.app j) x
  -/
  change Function.Injective (T.hom ≫ fun x j => G.π.app j x)
  have h : Function.Injective T.hom := by
    intro a b h
    suffices T.inv (T.hom a) = T.inv (T.hom b) by simpa
    rw [h]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    hE : CategoryTheory.Limits.IsLimit E := CategoryTheory.Limits.isLimitOfPreserv …
    G : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := CategoryT …
    hG : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.limitCone (F.c …
    T : CategoryTheory.Iso E.pt G.pt := hE.conePointUniqueUpToIso hG
    h : Function.Injective T.hom
    ⊢ Function.Injective (CategoryTheory.CategoryStruct.comp T.hom fun x j => G.π. …
  -/
  suffices Function.Injective fun (x : G.pt) j => G.π.app j x by exact this.comp h
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{t, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesLimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cone F
    hD : CategoryTheory.Limits.IsLimit D
    E : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := (Category …
    hE : CategoryTheory.Limits.IsLimit E := CategoryTheory.Limits.isLimitOfPreserv …
    G : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget C)) := CategoryT …
    hG : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.limitCone (F.c …
    T : CategoryTheory.Iso E.pt G.pt := hE.conePointUniqueUpToIso hG
    h : Function.Injective T.hom
    ⊢ Function.Injective fun x j => G.π.app j x
  -/
  apply Subtype.ext
  /-
    🎉 no goals
  -/


theorem isLimit_ext {D : Cone F} (hD : IsLimit D) (x y : D.pt) :
    (∀ j, D.π.app j x = D.π.app j y) → x = y := fun h =>
  Concrete.to_product_injective_of_isLimit _ hD (funext h)


theorem limit_ext [HasLimit F] (x y : ↑(limit F)) :
    (∀ j, limit.π F j x = limit.π F j y) → x = y :=
  Concrete.isLimit_ext F (limit.isLimit _) _ _


/--
Given surjections `⋯ ⟶ Xₙ₊₁ ⟶ Xₙ ⟶ ⋯ ⟶ X₀` in a concrete category whose forgetful functor
preserves sequential limits, the projection map `lim Xₙ ⟶ X₀` is surjective.
-/
lemma surjective_π_app_zero_of_surjective_map {C : Type u} [Category.{v} C] [ConcreteCategory.{v} C]
    [PreservesLimitsOfShape ℕᵒᵖ (forget C)] {F : ℕᵒᵖ ⥤ C} {c : Cone F}
    (hc : IsLimit c) (hF : ∀ n, Function.Surjective (F.map (homOfLE (Nat.le_succ n)).op)) :
    Function.Surjective (c.π.app ⟨0⟩) :=
  Types.surjective_π_app_zero_of_surjective_map (isLimitOfPreserves _ hc) hF


theorem from_union_surjective_of_isColimit {D : Cocone F} (hD : IsColimit D) :
    let ff : (Σj : J, F.obj j) → D.pt := fun a => D.ι.app a.1 a.2
    Function.Surjective ff := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    ⊢ let ff := fun a => (D.ι.app a.fst) a.snd;
      Function.Surjective ff
  -/
  intro ff x
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    ff : (Sigma fun j => (CategoryTheory.forget C).obj (F.obj j)) → (CategoryTheor …
    x : (CategoryTheory.forget C).obj D.pt
    ⊢ Exists fun a => Eq (ff a) x
  -/
  let E : Cocone (F ⋙ forget C) := (forget C).mapCocone D
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    ff : (Sigma fun j => (CategoryTheory.forget C).obj (F.obj j)) → (CategoryTheor …
    x : (CategoryTheory.forget C).obj D.pt
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    ⊢ Exists fun a => Eq (ff a) x
  -/
  let hE : IsColimit E := isColimitOfPreserves (forget C) hD
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    ff : (Sigma fun j => (CategoryTheory.forget C).obj (F.obj j)) → (CategoryTheor …
    x : (CategoryTheory.forget C).obj D.pt
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    hE : CategoryTheory.Limits.IsColimit E := CategoryTheory.Limits.isColimitOfPre …
    ⊢ Exists fun a => Eq (ff a) x
  -/
  obtain ⟨j, y, hy⟩ := Types.jointly_surjective_of_isColimit hE x
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    ff : (Sigma fun j => (CategoryTheory.forget C).obj (F.obj j)) → (CategoryTheor …
    x : (CategoryTheory.forget C).obj D.pt
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    hE : CategoryTheory.Limits.IsColimit E := CategoryTheory.Limits.isColimitOfPre …
    j : J
    y : (F.comp (CategoryTheory.forget C)).obj j
    hy : Eq (E.ι.app j y) x
    ⊢ Exists fun a => Eq (ff a) x
  -/
  exact ⟨⟨j, y⟩, hy⟩
  /-
    🎉 no goals
  -/


theorem isColimit_exists_rep {D : Cocone F} (hD : IsColimit D) (x : D.pt) :
    ∃ (j : J) (y : F.obj j), D.ι.app j y = x := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    x : (CategoryTheory.forget C).obj D.pt
    ⊢ Exists fun j => Exists fun y => Eq ((D.ι.app j) y) x
  -/
  obtain ⟨a, rfl⟩ := Concrete.from_union_surjective_of_isColimit F hD x
  /-
    case intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝¹ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    D : CategoryTheory.Limits.Cocone F
    hD : CategoryTheory.Limits.IsColimit D
    a : Sigma fun j => (CategoryTheory.forget C).obj (F.obj j)
    ⊢ Exists fun j => Exists fun y => Eq ((D.ι.app j) y) ((fun a => (D.ι.app a.fst …
  -/
  exact ⟨a.1, a.2, rfl⟩
  /-
    🎉 no goals
  -/


theorem colimit_exists_rep [HasColimit F] (x : ↑(colimit F)) :
    ∃ (j : J) (y : F.obj j), colimit.ι F j y = x :=
  Concrete.isColimit_exists_rep F (colimit.isColimit _) x


theorem isColimit_rep_eq_of_exists {D : Cocone F} {i j : J} (x : F.obj i) (y : F.obj j)
    (h : ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f x = F.map g y) :
    D.ι.app i x = D.ι.app j y := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    h : Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) x) ((F.map g …
    ⊢ Eq ((D.ι.app i) x) ((D.ι.app j) y)
  -/
  let E := (forget C).mapCocone D
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    h : Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) x) ((F.map g …
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    ⊢ Eq ((D.ι.app i) x) ((D.ι.app j) y)
  -/
  obtain ⟨k, f, g, (hfg : (F ⋙ forget C).map f x = F.map g y)⟩ := h
  /-
    case intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    k : J
    f : Quiver.Hom i k
    g : Quiver.Hom j k
    hfg : Eq ((F.comp (CategoryTheory.forget C)).map f x) ((F.map g) y)
    ⊢ Eq ((D.ι.app i) x) ((D.ι.app j) y)
  -/
  let h1 : (F ⋙ forget C).map f ≫ E.ι.app k = E.ι.app i := E.ι.naturality f
  /-
    case intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    k : J
    f : Quiver.Hom i k
    g : Quiver.Hom j k
    hfg : Eq ((F.comp (CategoryTheory.forget C)).map f x) ((F.map g) y)
    h1 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    ⊢ Eq ((D.ι.app i) x) ((D.ι.app j) y)
  -/
  let h2 : (F ⋙ forget C).map g ≫ E.ι.app k = E.ι.app j := E.ι.naturality g
  /-
    case intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    k : J
    f : Quiver.Hom i k
    g : Quiver.Hom j k
    hfg : Eq ((F.comp (CategoryTheory.forget C)).map f x) ((F.map g) y)
    h1 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    h2 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    ⊢ Eq ((D.ι.app i) x) ((D.ι.app j) y)
  -/
  show E.ι.app i x = E.ι.app j y
  /-
    case intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    k : J
    f : Quiver.Hom i k
    g : Quiver.Hom j k
    hfg : Eq ((F.comp (CategoryTheory.forget C)).map f x) ((F.map g) y)
    h1 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    h2 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    ⊢ Eq (E.ι.app i x) (E.ι.app j y)
  -/
  rw [← h1, types_comp_apply, hfg]
  /-
    case intro.intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    D : CategoryTheory.Limits.Cocone F
    i j : J
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    k : J
    f : Quiver.Hom i k
    g : Quiver.Hom j k
    hfg : Eq ((F.comp (CategoryTheory.forget C)).map f x) ((F.map g) y)
    h1 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    h2 : Eq (CategoryTheory.CategoryStruct.comp ((F.comp (CategoryTheory.forget C) …
    ⊢ Eq (E.ι.app k ((F.map g) y)) (E.ι.app j y)
  -/
  exact congrFun h2 y
  /-
    🎉 no goals
  -/


theorem colimit_rep_eq_of_exists [HasColimit F] {i j : J} (x : F.obj i) (y : F.obj j)
    (h : ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f x = F.map g y) :
    colimit.ι F i x = colimit.ι F j y :=
  Concrete.isColimit_rep_eq_of_exists F x y h


theorem isColimit_exists_of_rep_eq {D : Cocone F} {i j : J} (hD : IsColimit D)
    (x : F.obj i) (y : F.obj j) (h : D.ι.app _ x = D.ι.app _ y) :
    ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f x = F.map g y := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝² : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    inst✝ : CategoryTheory.IsFiltered J
    D : CategoryTheory.Limits.Cocone F
    i j : J
    hD : CategoryTheory.Limits.IsColimit D
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    h : Eq ((D.ι.app i) x) ((D.ι.app j) y)
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) x) ((F.map g) y)
  -/
  let E := (forget C).mapCocone D
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝² : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    inst✝ : CategoryTheory.IsFiltered J
    D : CategoryTheory.Limits.Cocone F
    i j : J
    hD : CategoryTheory.Limits.IsColimit D
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    h : Eq ((D.ι.app i) x) ((D.ι.app j) y)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) x) ((F.map g) y)
  -/
  let hE : IsColimit E := isColimitOfPreserves _ hD
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    J : Type w
    inst✝² : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget C)
    inst✝ : CategoryTheory.IsFiltered J
    D : CategoryTheory.Limits.Cocone F
    i j : J
    hD : CategoryTheory.Limits.IsColimit D
    x : (CategoryTheory.forget C).obj (F.obj i)
    y : (CategoryTheory.forget C).obj (F.obj j)
    h : Eq ((D.ι.app i) x) ((D.ι.app j) y)
    E : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.forget C)) := (Catego …
    hE : CategoryTheory.Limits.IsColimit E := CategoryTheory.Limits.isColimitOfPre …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) x) ((F.map g) y)
  -/
  exact (Types.FilteredColimit.isColimit_eq_iff (F ⋙ forget C) hE).mp h
  /-
    🎉 no goals
  -/


theorem isColimit_rep_eq_iff_exists {D : Cocone F} {i j : J} (hD : IsColimit D)
    (x : F.obj i) (y : F.obj j) :
    D.ι.app i x = D.ι.app j y ↔ ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f x = F.map g y :=
  ⟨Concrete.isColimit_exists_of_rep_eq.{t} _ hD _ _,
   Concrete.isColimit_rep_eq_of_exists _ _ _⟩


theorem colimit_exists_of_rep_eq [HasColimit F] {i j : J} (x : F.obj i) (y : F.obj j)
    (h : colimit.ι F _ x = colimit.ι F _ y) :
    ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f x = F.map g y :=
  Concrete.isColimit_exists_of_rep_eq.{t} F (colimit.isColimit _) x y h


theorem colimit_rep_eq_iff_exists [HasColimit F] {i j : J} (x : F.obj i) (y : F.obj j) :
    colimit.ι F i x = colimit.ι F j y ↔ ∃ (k : _) (f : i ⟶ k) (g : j ⟶ k), F.map f x = F.map g y :=
  ⟨Concrete.colimit_exists_of_rep_eq.{t} _ _ _, Concrete.colimit_rep_eq_of_exists _ _ _⟩


