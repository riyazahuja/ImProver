/-- The equivalence `(forget C).obj (∏ᶜ F) ≃ ∀ j, F j` if `F : J → C` is a family of objects
in a concrete category `C`. -/
noncomputable def productEquiv : (forget C).obj (∏ᶜ F) ≃ ∀ j, F j :=
  ((PreservesProduct.iso (forget C) F) ≪≫ (Types.productIso.{w, v} (fun j => F j))).toEquiv


@[simp]
lemma productEquiv_apply_apply (x : (forget C).obj (∏ᶜ F)) (j : J) :
    productEquiv F x j = Pi.π F j x :=
  congr_fun (piComparison_comp_π (forget C) F j) x


@[simp]
lemma productEquiv_symm_apply_π (x : ∀ j, F j) (j : J) :
    Pi.π F j ((productEquiv F).symm x) = x j := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    J : Type w
    F : J → C
    inst✝¹ : CategoryTheory.Limits.HasProduct F
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    x : (j : J) → (CategoryTheory.forget C).obj (F j)
    j : J
    ⊢ Eq ((CategoryTheory.Limits.Pi.π F j) ((CategoryTheory.Limits.Concrete.produc …
  -/
  rw [← productEquiv_apply_apply, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma Pi.map_ext (x y : F.obj (∏ᶜ f : C))
    (h : ∀ i, F.map (Pi.π f i) x = F.map (Pi.π f i) y) : x = y := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : Type w
    f : J → C
    inst✝⁶ : CategoryTheory.Limits.HasProduct f
    D : Type t
    inst✝⁵ : CategoryTheory.Category.{r, t} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    F : CategoryTheory.Functor C D
    inst✝³ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor …
    inst✝² : CategoryTheory.Limits.HasProduct fun j => F.obj (f j)
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    x y : (CategoryTheory.forget D).obj (F.obj (CategoryTheory.Limits.piObj f))
    h : ∀ (i : J), Eq ((F.map (CategoryTheory.Limits.Pi.π f i)) x) ((F.map (Catego …
    ⊢ Eq x y
  -/
  apply ConcreteCategory.injective_of_mono_of_preservesPullback (PreservesProduct.iso F f).hom
  apply @Concrete.limit_ext.{w, w, r, t} D
    _ _ (Discrete J) _ _ _ _ (piComparison F _ x) (piComparison F _ y)
  /-
    case a
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : Type w
    f : J → C
    inst✝⁶ : CategoryTheory.Limits.HasProduct f
    D : Type t
    inst✝⁵ : CategoryTheory.Category.{r, t} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    F : CategoryTheory.Functor C D
    inst✝³ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor …
    inst✝² : CategoryTheory.Limits.HasProduct fun j => F.obj (f j)
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    x y : (CategoryTheory.forget D).obj (F.obj (CategoryTheory.Limits.piObj f))
    h : ∀ (i : J), Eq ((F.map (CategoryTheory.Limits.Pi.π f i)) x) ((F.map (Catego …
    ⊢ ∀ (j : CategoryTheory.Discrete J), Eq ((CategoryTheory.Limits.limit.π (Categ …
  -/
  intro ⟨(j : J)⟩
  show ((forget D).map (piComparison F f) ≫ (forget D).map (limit.π _ _)) x =
    ((forget D).map (piComparison F f) ≫ (forget D).map _) y
  /-
    case a
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : Type w
    f : J → C
    inst✝⁶ : CategoryTheory.Limits.HasProduct f
    D : Type t
    inst✝⁵ : CategoryTheory.Category.{r, t} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    F : CategoryTheory.Functor C D
    inst✝³ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor …
    inst✝² : CategoryTheory.Limits.HasProduct fun j => F.obj (f j)
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    x y : (CategoryTheory.forget D).obj (F.obj (CategoryTheory.Limits.piObj f))
    h : ∀ (i : J), Eq ((F.map (CategoryTheory.Limits.Pi.π f i)) x) ((F.map (Catego …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.forget D).map (Categ …
  -/
  rw [← (forget D).map_comp, piComparison_comp_π]
  /-
    case a
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    J : Type w
    f : J → C
    inst✝⁶ : CategoryTheory.Limits.HasProduct f
    D : Type t
    inst✝⁵ : CategoryTheory.Category.{r, t} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    F : CategoryTheory.Functor C D
    inst✝³ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor …
    inst✝² : CategoryTheory.Limits.HasProduct fun j => F.obj (f j)
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    x y : (CategoryTheory.forget D).obj (F.obj (CategoryTheory.Limits.piObj f))
    h : ∀ (i : J), Eq ((F.map (CategoryTheory.Limits.Pi.π f i)) x) ((F.map (Catego …
    j : J
    ⊢ Eq ((CategoryTheory.forget D).map (F.map (CategoryTheory.Limits.Pi.π f j)) x …
  -/
  exact h j
  /-
    🎉 no goals
  -/


/-- If `forget C` preserves terminals and `X` is terminal, then `(forget C).obj X` is a
singleton. -/
noncomputable def uniqueOfTerminalOfPreserves [PreservesLimit (Functor.empty.{0} C) (forget C)]
    (X : C) (h : IsTerminal X) : Unique ((forget C).obj X) :=
  Types.isTerminalEquivUnique ((forget C).obj X) <| IsTerminal.isTerminalObj (forget C) X h


/-- If `forget C` reflects terminals and `(forget C).obj X` is a singleton, then `X` is terminal. -/
noncomputable def terminalOfUniqueOfReflects [ReflectsLimit (Functor.empty.{0} C) (forget C)]
    (X : C) (h : Unique ((forget C).obj X)) : IsTerminal X :=
  IsTerminal.isTerminalOfObj (forget C) X <| (Types.isTerminalEquivUnique ((forget C).obj X)).symm h


/-- The equivalence `IsTerminal X ≃ Unique ((forget C).obj X)` if the forgetful functor
preserves and reflects terminals. -/
noncomputable def terminalIffUnique [PreservesLimit (Functor.empty.{0} C) (forget C)]
    [ReflectsLimit (Functor.empty.{0} C) (forget C)] (X : C) :
    IsTerminal X ≃ Unique ((forget C).obj X) :=
  (IsTerminal.isTerminalIffObj (forget C) X).trans <| Types.isTerminalEquivUnique _


/-- The equivalence `(forget C).obj (⊤_ C) ≃ PUnit` when `C` is a concrete category. -/
noncomputable def terminalEquiv : (forget C).obj (⊤_ C) ≃ PUnit :=
  (PreservesTerminal.iso (forget C) ≪≫ Types.terminalIso).toEquiv


noncomputable instance : Unique ((forget C).obj (⊤_ C)) where
  default := (terminalEquiv C).symm PUnit.unit
  uniq _ := (terminalEquiv C).injective (Subsingleton.elim _ _)


/-- If `forget C` preserves initials and `X` is initial, then `(forget C).obj X` is empty. -/
lemma empty_of_initial_of_preserves [PreservesColimit (Functor.empty.{0} C) (forget C)] (X : C)
    (h : Nonempty (IsInitial X)) : IsEmpty ((forget C).obj X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
    X : C
    h : Nonempty (CategoryTheory.Limits.IsInitial X)
    ⊢ IsEmpty ((CategoryTheory.forget C).obj X)
  -/
  rw [← Types.initial_iff_empty]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
    X : C
    h : Nonempty (CategoryTheory.Limits.IsInitial X)
    ⊢ Nonempty (CategoryTheory.Limits.IsInitial ((CategoryTheory.forget C).obj X))
  -/
  exact Nonempty.map (IsInitial.isInitialObj (forget C) _) h
  /-
    🎉 no goals
  -/


/-- If `forget C` reflects initials and `(forget C).obj X` is empty, then `X` is initial. -/
lemma initial_of_empty_of_reflects [ReflectsColimit (Functor.empty.{0} C) (forget C)] (X : C)
    (h : IsEmpty ((forget C).obj X)) : Nonempty (IsInitial X) :=
  Nonempty.map (IsInitial.isInitialOfObj (forget C) _) <|
    (Types.initial_iff_empty ((forget C).obj X)).mpr h


/-- If `forget C` preserves and reflects initials, then `X` is initial if and only if
`(forget C).obj X` is empty. -/
lemma initial_iff_empty_of_preserves_of_reflects [PreservesColimit (Functor.empty.{0} C) (forget C)]
    [ReflectsColimit (Functor.empty.{0} C) (forget C)] (X : C) :
    Nonempty (IsInitial X) ↔ IsEmpty ((forget C).obj X) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty  …
    inst✝ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Functor.empty C) …
    X : C
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsInitial X)) (IsEmpty ((CategoryTheory …
  -/
  rw [← Types.initial_iff_empty, (IsInitial.isInitialIffObj (forget C) X).nonempty_congr]
  /-
    🎉 no goals
  -/


/-- The equivalence `(forget C).obj (X₁ ⨯ X₂) ≃ ((forget C).obj X₁) × ((forget C).obj X₂)`
if `X₁` and `X₂` are objects in a concrete category `C`. -/
noncomputable def prodEquiv : (forget C).obj (X₁ ⨯ X₂) ≃ X₁ × X₂ :=
  (PreservesLimitPair.iso (forget C) X₁ X₂ ≪≫ Types.binaryProductIso _ _).toEquiv


@[simp]
lemma prodEquiv_apply_fst (x : (forget C).obj (X₁ ⨯ X₂)) :
    (prodEquiv X₁ X₂ x).fst = (Limits.prod.fst : X₁ ⨯ X₂ ⟶ X₁) x :=
  congr_fun (prodComparison_fst (forget C) X₁ X₂) x


@[simp]
lemma prodEquiv_apply_snd (x : (forget C).obj (X₁ ⨯ X₂)) :
    (prodEquiv X₁ X₂ x).snd = (Limits.prod.snd : X₁ ⨯ X₂ ⟶ X₂) x :=
  congr_fun (prodComparison_snd (forget C) X₁ X₂) x


@[simp]
lemma prodEquiv_symm_apply_fst (x : X₁ × X₂) :
    (Limits.prod.fst : X₁ ⨯ X₂ ⟶ X₁) ((prodEquiv X₁ X₂).symm x) = x.1 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X₁ X₂ : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X₁ X₂
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂ …
    x : Prod ((CategoryTheory.forget C).obj X₁) ((CategoryTheory.forget C).obj X₂)
    ⊢ Eq (CategoryTheory.Limits.prod.fst ((CategoryTheory.Limits.Concrete.prodEqui …
  -/
  obtain ⟨y, rfl⟩ := (prodEquiv X₁ X₂).surjective x
  /-
    case intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X₁ X₂ : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X₁ X₂
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂ …
    y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.prod X₁ X₂)
    ⊢ Eq (CategoryTheory.Limits.prod.fst ((CategoryTheory.Limits.Concrete.prodEqui …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma prodEquiv_symm_apply_snd (x : X₁ × X₂) :
    (Limits.prod.snd : X₁ ⨯ X₂ ⟶ X₂) ((prodEquiv X₁ X₂).symm x) = x.2 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X₁ X₂ : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X₁ X₂
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂ …
    x : Prod ((CategoryTheory.forget C).obj X₁) ((CategoryTheory.forget C).obj X₂)
    ⊢ Eq (CategoryTheory.Limits.prod.snd ((CategoryTheory.Limits.Concrete.prodEqui …
  -/
  obtain ⟨y, rfl⟩ := (prodEquiv X₁ X₂).surjective x
  /-
    case intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X₁ X₂ : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X₁ X₂
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂ …
    y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.prod X₁ X₂)
    ⊢ Eq (CategoryTheory.Limits.prod.snd ((CategoryTheory.Limits.Concrete.prodEqui …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- In a concrete category `C`, given two morphisms `f₁ : X₁ ⟶ S` and `f₂ : X₂ ⟶ S`,
the elements in `pullback f₁ f₁` can be identified to compatible tuples of
elements in `X₁` and `X₂`. -/
noncomputable def pullbackEquiv :
    (forget C).obj (pullback f₁ f₂) ≃ { p : X₁ × X₂ // f₁ p.1 = f₂ p.2 } :=
  (PreservesPullback.iso (forget C) f₁ f₂ ≪≫
    Types.pullbackIsoPullback ((forget C).map f₁) ((forget C).map f₂)).toEquiv


/-- Constructor for elements in a pullback in a concrete category. -/
noncomputable def pullbackMk (x₁ : X₁) (x₂ : X₂) (h : f₁ x₁ = f₂ x₂) :
    (forget C).obj (pullback f₁ f₂) :=
  (pullbackEquiv f₁ f₂).symm ⟨⟨x₁, x₂⟩, h⟩


lemma pullbackMk_surjective (x : (forget C).obj (pullback f₁ f₂)) :
    ∃ (x₁ : X₁) (x₂ : X₂) (h : f₁ x₁ = f₂ x₂), x = pullbackMk f₁ f₂ x₁ x₂ h := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X₁ X₂ S : C
    f₁ : Quiver.Hom X₁ S
    f₂ : Quiver.Hom X₂ S
    inst✝¹ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f₁  …
    x : (CategoryTheory.forget C).obj (CategoryTheory.Limits.pullback f₁ f₂)
    ⊢ Exists fun x₁ => Exists fun x₂ => Exists fun h => Eq x (CategoryTheory.Limit …
  -/
  obtain ⟨⟨⟨x₁, x₂⟩, h⟩, rfl⟩ := (pullbackEquiv f₁ f₂).symm.surjective x
  /-
    case intro.mk.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X₁ X₂ S : C
    f₁ : Quiver.Hom X₁ S
    f₂ : Quiver.Hom X₂ S
    inst✝¹ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f₁  …
    x₁ : (CategoryTheory.forget C).obj X₁
    x₂ : (CategoryTheory.forget C).obj X₂
    h : Eq (f₁ { fst := x₁, snd := x₂ }.1) (f₂ { fst := x₁, snd := x₂ }.2)
    ⊢ Exists fun x₁_1 => Exists fun x₂_1 => Exists fun h_1 => Eq ((CategoryTheory. …
  -/
  exact ⟨x₁, x₂, h, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma pullbackMk_fst (x₁ : X₁) (x₂ : X₂) (h : f₁ x₁ = f₂ x₂) :
    pullback.fst f₁ f₂ (pullbackMk f₁ f₂ x₁ x₂ h) = x₁ :=
  (congr_fun (PreservesPullback.iso_inv_fst (forget C) f₁ f₂) _).trans
    (congr_fun (Types.pullbackIsoPullback_inv_fst ((forget C).map f₁) ((forget C).map f₂)) _)


@[simp]
lemma pullbackMk_snd (x₁ : X₁) (x₂ : X₂) (h : f₁ x₁ = f₂ x₂) :
    pullback.snd f₁ f₂ (pullbackMk f₁ f₂ x₁ x₂ h) = x₂ :=
  (congr_fun (PreservesPullback.iso_inv_snd (forget C) f₁ f₂) _).trans
    (congr_fun (Types.pullbackIsoPullback_inv_snd ((forget C).map f₁) ((forget C).map f₂)) _)


theorem widePullback_ext {B : C} {ι : Type w} {X : ι → C} (f : ∀ j : ι, X j ⟶ B)
    [HasWidePullback B X f] [PreservesLimit (wideCospan B X f) (forget C)]
    (x y : ↑(widePullback B X f)) (h₀ : base f x = base f y) (h : ∀ j, π f j x = π f j y) :
    x = y := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    B : C
    ι : Type w
    X : ι → C
    f : (j : ι) → Quiver.Hom (X j) B
    inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
    h₀ : Eq ((CategoryTheory.Limits.WidePullback.base f) x) ((CategoryTheory.Limit …
    h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
    ⊢ Eq x y
  -/
  apply Concrete.limit_ext
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    B : C
    ι : Type w
    X : ι → C
    f : (j : ι) → Quiver.Hom (X j) B
    inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
    h₀ : Eq ((CategoryTheory.Limits.WidePullback.base f) x) ((CategoryTheory.Limit …
    h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
    ⊢ ∀ (j : CategoryTheory.Limits.WidePullbackShape ι), Eq ((CategoryTheory.Limit …
  -/
  rintro (_ | j)
    /-
      case a.none
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      ι : Type w
      X : ι → C
      f : (j : ι) → Quiver.Hom (X j) B
      inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
      x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
      h₀ : Eq ((CategoryTheory.Limits.WidePullback.base f) x) ((CategoryTheory.Limit …
      h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
      ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.WidePullbackShape. …
    -/
  · exact h₀
    /-
      🎉 no goals
    -/
    /-
      case a.some
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      ι : Type w
      X : ι → C
      f : (j : ι) → Quiver.Hom (X j) B
      inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
      x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
      h₀ : Eq ((CategoryTheory.Limits.WidePullback.base f) x) ((CategoryTheory.Limit …
      h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
      j : ι
      ⊢ Eq ((CategoryTheory.Limits.limit.π (CategoryTheory.Limits.WidePullbackShape. …
    -/
  · apply h
    /-
      🎉 no goals
    -/


theorem widePullback_ext' {B : C} {ι : Type w} [Nonempty ι] {X : ι → C}
    (f : ∀ j : ι, X j ⟶ B) [HasWidePullback.{w} B X f]
    [PreservesLimit (wideCospan B X f) (forget C)] (x y : ↑(widePullback B X f))
    (h : ∀ j, π f j x = π f j y) : x = y := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    B : C
    ι : Type w
    inst✝² : Nonempty ι
    X : ι → C
    f : (j : ι) → Quiver.Hom (X j) B
    inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
    h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
    ⊢ Eq x y
  -/
  apply Concrete.widePullback_ext _ _ _ _ h
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    B : C
    ι : Type w
    inst✝² : Nonempty ι
    X : ι → C
    f : (j : ι) → Quiver.Hom (X j) B
    inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
    h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
    ⊢ Eq ((CategoryTheory.Limits.WidePullback.base f) x) ((CategoryTheory.Limits.W …
  -/
  inhabit ι
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    B : C
    ι : Type w
    inst✝² : Nonempty ι
    X : ι → C
    f : (j : ι) → Quiver.Hom (X j) B
    inst✝¹ : CategoryTheory.Limits.HasWidePullback B X f
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.WidePullba …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePullback B X f)
    h : ∀ (j : ι), Eq ((CategoryTheory.Limits.WidePullback.π f j) x) ((CategoryThe …
    inhabited_h : Inhabited ι
    ⊢ Eq ((CategoryTheory.Limits.WidePullback.base f) x) ((CategoryTheory.Limits.W …
  -/
  simp only [← π_arrow f default, comp_apply, h]
  /-
    🎉 no goals
  -/


theorem multiequalizer_ext {I : MulticospanIndex.{w, w'} C} [HasMultiequalizer I]
    [PreservesLimit I.multicospan (forget C)] (x y : ↑(multiequalizer I))
    (h : ∀ t : I.L, Multiequalizer.ι I t x = Multiequalizer.ι I t y) : x = y := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    I : CategoryTheory.Limits.MulticospanIndex C
    inst✝¹ : CategoryTheory.Limits.HasMultiequalizer I
    inst✝ : CategoryTheory.Limits.PreservesLimit I.multicospan (CategoryTheory.for …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.multiequalizer I)
    h : ∀ (t : I.L), Eq ((CategoryTheory.Limits.Multiequalizer.ι I t) x) ((Categor …
    ⊢ Eq x y
  -/
  apply Concrete.limit_ext
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    I : CategoryTheory.Limits.MulticospanIndex C
    inst✝¹ : CategoryTheory.Limits.HasMultiequalizer I
    inst✝ : CategoryTheory.Limits.PreservesLimit I.multicospan (CategoryTheory.for …
    x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.multiequalizer I)
    h : ∀ (t : I.L), Eq ((CategoryTheory.Limits.Multiequalizer.ι I t) x) ((Categor …
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingMulticospan I.fstTo I.sndTo), Eq ((Categ …
  -/
  rintro (a | b)
    /-
      case a.left
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      inst✝¹ : CategoryTheory.Limits.HasMultiequalizer I
      inst✝ : CategoryTheory.Limits.PreservesLimit I.multicospan (CategoryTheory.for …
      x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.multiequalizer I)
      h : ∀ (t : I.L), Eq ((CategoryTheory.Limits.Multiequalizer.ι I t) x) ((Categor …
      a : I.L
      ⊢ Eq ((CategoryTheory.Limits.limit.π I.multicospan (CategoryTheory.Limits.Walk …
    -/
  · apply h
    /-
      🎉 no goals
    -/
    /-
      case a.right
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      inst✝¹ : CategoryTheory.Limits.HasMultiequalizer I
      inst✝ : CategoryTheory.Limits.PreservesLimit I.multicospan (CategoryTheory.for …
      x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.multiequalizer I)
      h : ∀ (t : I.L), Eq ((CategoryTheory.Limits.Multiequalizer.ι I t) x) ((Categor …
      b : I.R
      ⊢ Eq ((CategoryTheory.Limits.limit.π I.multicospan (CategoryTheory.Limits.Walk …
    -/
  · rw [← limit.w I.multicospan (WalkingMulticospan.Hom.fst b), comp_apply, comp_apply]
    /-
      case a.right
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      inst✝¹ : CategoryTheory.Limits.HasMultiequalizer I
      inst✝ : CategoryTheory.Limits.PreservesLimit I.multicospan (CategoryTheory.for …
      x y : (CategoryTheory.forget C).obj (CategoryTheory.Limits.multiequalizer I)
      h : ∀ (t : I.L), Eq ((CategoryTheory.Limits.Multiequalizer.ι I t) x) ((Categor …
      b : I.R
      ⊢ Eq ((I.multicospan.map (CategoryTheory.Limits.WalkingMulticospan.Hom.fst b)) …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


/-- An auxiliary equivalence to be used in `multiequalizerEquiv` below. -/
def multiequalizerEquivAux (I : MulticospanIndex.{w, w'} C) :
    (I.multicospan ⋙ forget C).sections ≃
    { x : ∀ i : I.L, I.left i // ∀ i : I.R, I.fst i (x _) = I.snd i (x _) } where
  toFun x :=
    ⟨fun _ => x.1 (WalkingMulticospan.left _), fun i => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        i : I.R
        ⊢ Eq ((I.fst i) ((fun x_1 => ↑x (CategoryTheory.Limits.WalkingMulticospan.left …
      -/
      have a := x.2 (WalkingMulticospan.Hom.fst i)
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        i : I.R
        a : Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Lim …
        ⊢ Eq ((I.fst i) ((fun x_1 => ↑x (CategoryTheory.Limits.WalkingMulticospan.left …
      -/
      have b := x.2 (WalkingMulticospan.Hom.snd i)
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        i : I.R
        a : Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Lim …
        b : Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Lim …
        ⊢ Eq ((I.fst i) ((fun x_1 => ↑x (CategoryTheory.Limits.WalkingMulticospan.left …
      -/
      rw [← b] at a
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        i : I.R
        a : Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Lim …
        b : Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Lim …
        ⊢ Eq ((I.fst i) ((fun x_1 => ↑x (CategoryTheory.Limits.WalkingMulticospan.left …
      -/
      exact a⟩
      /-
        🎉 no goals
      -/
  invFun x :=
    { val := fun j =>
        match j with
        | WalkingMulticospan.left _ => x.1 _
        | WalkingMulticospan.right b => I.fst b (x.1 _)
      property := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.ConcreteCategory C
          I : CategoryTheory.Limits.MulticospanIndex C
          x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
          ⊢ Membership.mem (I.multicospan.comp (CategoryTheory.forget C)).sections fun j …
        -/
        rintro (a | b) (a' | b') (f | f | f)
          /-
            case left.left.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.ConcreteCategory C
            I : CategoryTheory.Limits.MulticospanIndex C
            x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
            a : I.L
            ⊢ Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Limit …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case left.right.fst
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.ConcreteCategory C
            I : CategoryTheory.Limits.MulticospanIndex C
            x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
            b' : I.R
            ⊢ Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Limit …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case left.right.snd
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.ConcreteCategory C
            I : CategoryTheory.Limits.MulticospanIndex C
            x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
            b' : I.R
            ⊢ Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Limit …
          -/
        · dsimp
          /-
            case left.right.snd
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.ConcreteCategory C
            I : CategoryTheory.Limits.MulticospanIndex C
            x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
            b' : I.R
            ⊢ Eq ((CategoryTheory.forget C).map (I.snd b') (↑x (I.sndTo b'))) ((I.fst b')  …
          -/
          exact (x.2 b').symm
          /-
            🎉 no goals
          -/
          /-
            case right.right.id
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.ConcreteCategory C
            I : CategoryTheory.Limits.MulticospanIndex C
            x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
            b : I.R
            ⊢ Eq ((I.multicospan.comp (CategoryTheory.forget C)).map (CategoryTheory.Limit …
          -/
        · simp }
          /-
            🎉 no goals
          -/
  left_inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      ⊢ Function.LeftInverse (fun x => ⟨fun j => CategoryTheory.Limits.Concrete.mult …
    -/
    intro x; ext (a | b)
      /-
        case a.h.left
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        a : I.L
        ⊢ Eq (↑((fun x => ⟨fun j => CategoryTheory.Limits.Concrete.multiequalizerEquiv …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a.h.right
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        b : I.R
        ⊢ Eq (↑((fun x => ⟨fun j => CategoryTheory.Limits.Concrete.multiequalizerEquiv …
      -/
    · rw [← x.2 (WalkingMulticospan.Hom.fst b)]
      /-
        case a.h.right
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.ConcreteCategory C
        I : CategoryTheory.Limits.MulticospanIndex C
        x : ↑(I.multicospan.comp (CategoryTheory.forget C)).sections
        b : I.R
        ⊢ Eq (↑((fun x => ⟨fun j => CategoryTheory.Limits.Concrete.multiequalizerEquiv …
      -/
      rfl
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      ⊢ Function.RightInverse (fun x => ⟨fun j => CategoryTheory.Limits.Concrete.mul …
    -/
    intro x
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
      ⊢ Eq ((fun x => ⟨fun x_1 => ↑x (CategoryTheory.Limits.WalkingMulticospan.left  …
    -/
    ext i
    /-
      case a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      I : CategoryTheory.Limits.MulticospanIndex C
      x : Subtype fun x => ∀ (i : I.R), Eq ((I.fst i) (x (I.fstTo i))) ((I.snd i) (x …
      i : I.L
      ⊢ Eq (↑((fun x => ⟨fun x_1 => ↑x (CategoryTheory.Limits.WalkingMulticospan.lef …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The equivalence between the noncomputable multiequalizer and
the concrete multiequalizer. -/
noncomputable def multiequalizerEquiv (I : MulticospanIndex.{w, w'} C) [HasMultiequalizer I]
    [PreservesLimit I.multicospan (forget C)] :
    (multiequalizer I : C) ≃
      { x : ∀ i : I.L, I.left i // ∀ i : I.R, I.fst i (x _) = I.snd i (x _) } :=
  letI h1 := limit.isLimit I.multicospan
  letI h2 := isLimitOfPreserves (forget C) h1
  letI E := h2.conePointUniqueUpToIso (Types.limitConeIsLimit.{max w w', v} _)
  Equiv.trans E.toEquiv (Concrete.multiequalizerEquivAux I)


@[simp]
theorem multiequalizerEquiv_apply (I : MulticospanIndex.{w, w'} C) [HasMultiequalizer I]
    [PreservesLimit I.multicospan (forget C)] (x : ↑(multiequalizer I)) (i : I.L) :
    ((Concrete.multiequalizerEquiv I) x : ∀ i : I.L, I.left i) i = Multiequalizer.ι I i x :=
  rfl


theorem widePushout_exists_rep {B : C} {α : Type _} {X : α → C} (f : ∀ j : α, B ⟶ X j)
    [HasWidePushout.{v} B X f] [PreservesColimit (wideSpan B X f) (forget C)]
    (x : ↑(widePushout B X f)) : (∃ y : B, head f y = x) ∨ ∃ (i : α) (y : X i), ι f i y = x := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    B : C
    α : Type v
    X : α → C
    f : (j : α) → Quiver.Hom B (X j)
    inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
    x : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePushout B X f)
    ⊢ Or (Exists fun y => Eq ((CategoryTheory.Limits.WidePushout.head f) y) x) (Ex …
  -/
  obtain ⟨_ | j, y, rfl⟩ := Concrete.colimit_exists_rep _ x
    /-
      case intro.none.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      y : (CategoryTheory.forget C).obj ((CategoryTheory.Limits.WidePushoutShape.wid …
      ⊢ Or (Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.head f) y_1) (( …
    -/
  · left
    /-
      case intro.none.intro.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      y : (CategoryTheory.forget C).obj ((CategoryTheory.Limits.WidePushoutShape.wid …
      ⊢ Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.head f) y_1) ((Cate …
    -/
    use y
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      y : (CategoryTheory.forget C).obj ((CategoryTheory.Limits.WidePushoutShape.wid …
      ⊢ Eq ((CategoryTheory.Limits.WidePushout.head f) y) ((CategoryTheory.Limits.co …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.some.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      j : α
      y : (CategoryTheory.forget C).obj ((CategoryTheory.Limits.WidePushoutShape.wid …
      ⊢ Or (Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.head f) y_1) (( …
    -/
  · right
    /-
      case intro.some.intro.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      j : α
      y : (CategoryTheory.forget C).obj ((CategoryTheory.Limits.WidePushoutShape.wid …
      ⊢ Exists fun i => Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.ι f …
    -/
    use j, y
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      j : α
      y : (CategoryTheory.forget C).obj ((CategoryTheory.Limits.WidePushoutShape.wid …
      ⊢ Eq ((CategoryTheory.Limits.WidePushout.ι f j) y) ((CategoryTheory.Limits.col …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem widePushout_exists_rep' {B : C} {α : Type _} [Nonempty α] {X : α → C}
    (f : ∀ j : α, B ⟶ X j) [HasWidePushout.{v} B X f] [PreservesColimit (wideSpan B X f) (forget C)]
    (x : ↑(widePushout B X f)) : ∃ (i : α) (y : X i), ι f i y = x := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    B : C
    α : Type v
    inst✝² : Nonempty α
    X : α → C
    f : (j : α) → Quiver.Hom B (X j)
    inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
    x : (CategoryTheory.forget C).obj (CategoryTheory.Limits.widePushout B X f)
    ⊢ Exists fun i => Exists fun y => Eq ((CategoryTheory.Limits.WidePushout.ι f i …
  -/
  rcases Concrete.widePushout_exists_rep f x with (⟨y, rfl⟩ | ⟨i, y, rfl⟩)
    /-
      case inl.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      inst✝² : Nonempty α
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      y : (CategoryTheory.forget C).obj B
      ⊢ Exists fun i => Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.ι f …
    -/
  · inhabit α
    /-
      case inl.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      inst✝² : Nonempty α
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      y : (CategoryTheory.forget C).obj B
      inhabited_h : Inhabited α
      ⊢ Exists fun i => Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.ι f …
    -/
    use default, f _ y
    /-
      case h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      inst✝² : Nonempty α
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      y : (CategoryTheory.forget C).obj B
      inhabited_h : Inhabited α
      ⊢ Eq ((CategoryTheory.Limits.WidePushout.ι f Inhabited.default) ((f Inhabited. …
    -/
    simp only [← arrow_ι _ default, comp_apply]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.ConcreteCategory C
      B : C
      α : Type v
      inst✝² : Nonempty α
      X : α → C
      f : (j : α) → Quiver.Hom B (X j)
      inst✝¹ : CategoryTheory.Limits.HasWidePushout B X f
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.WidePush …
      i : α
      y : (CategoryTheory.forget C).obj (X i)
      ⊢ Exists fun i_1 => Exists fun y_1 => Eq ((CategoryTheory.Limits.WidePushout.ι …
    -/
  · use i, y
    /-
      🎉 no goals
    -/


theorem cokernel_funext {C : Type*} [Category C] [HasZeroMorphisms C] [ConcreteCategory C]
    {M N K : C} {f : M ⟶ N} [HasCokernel f] {g h : cokernel f ⟶ K}
    (w : ∀ n : N, g (cokernel.π f n) = h (cokernel.π f n)) : g = h := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    M N K : C
    f : Quiver.Hom M N
    inst✝ : CategoryTheory.Limits.HasCokernel f
    g h : Quiver.Hom (CategoryTheory.Limits.cokernel f) K
    w : ∀ (n : (CategoryTheory.forget C).obj N), Eq (g ((CategoryTheory.Limits.cok …
    ⊢ Eq g h
  -/
  ext x
  /-
    case h.w
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.ConcreteCategory C
    M N K : C
    f : Quiver.Hom M N
    inst✝ : CategoryTheory.Limits.HasCokernel f
    g h : Quiver.Hom (CategoryTheory.Limits.cokernel f) K
    w : ∀ (n : (CategoryTheory.forget C).obj N), Eq (g ((CategoryTheory.Limits.cok …
    x : (CategoryTheory.forget C).obj N
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π …
  -/
  simpa using w x
  /-
    🎉 no goals
  -/

-- TODO: Add analogous lemmas about coproducts and coequalizers.


