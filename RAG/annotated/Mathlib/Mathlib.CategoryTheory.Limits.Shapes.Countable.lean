/--
A category has all countable limits if every functor `J ⥤ C` with a `CountableCategory J`
instance and `J : Type` has a limit.
-/
class HasCountableLimits : Prop where
  /-- `C` has all limits over any type `J` whose objects and morphisms lie in the same universe
  and which has countably many objects and morphisms -/
  out (J : Type) [SmallCategory J] [CountableCategory J] : HasLimitsOfShape J C


instance (priority := 100) hasFiniteLimits_of_hasCountableLimits [HasCountableLimits C] :
    HasFiniteLimits C where
  out J := HasCountableLimits.out J


instance (priority := 100) hasCountableLimits_of_hasLimits [HasLimits C] :
    HasCountableLimits C where
  out := inferInstance


universe v in
instance [Category.{v} J] [CountableCategory J] [HasCountableLimits C] : HasLimitsOfShape J C :=
  have : HasLimitsOfShape (HomAsType J) C := HasCountableLimits.out (HomAsType J)
  hasLimitsOfShape_of_equivalence (homAsTypeEquiv J)


/-- A category has countable products if it has all products indexed by countable types. -/
class HasCountableProducts where
  out (J : Type) [Countable J] : HasProductsOfShape J C


instance [HasCountableProducts C] : HasProductsOfShape J C :=
  have : Countable (Shrink.{0} J) := Countable.of_equiv _ (equivShrink.{0} J)
  have : HasLimitsOfShape (Discrete (Shrink.{0} J)) C := HasCountableProducts.out _
  hasLimitsOfShape_of_equivalence (Discrete.equivalence (equivShrink.{0} J)).symm


instance (priority := 100) hasCountableProducts_of_hasProducts [HasProducts C] :
    HasCountableProducts C where
  out _ :=
    have : HasProducts.{0} C := has_smallest_products_of_hasProducts
    inferInstance


instance (priority := 100) hasCountableProducts_of_hasCountableLimits [HasCountableLimits C] :
    HasCountableProducts C where
  out _ := inferInstance


instance (priority := 100) hasFiniteProducts_of_hasCountableProducts [HasCountableProducts C] :
    HasFiniteProducts C where
  out _ := inferInstance


/--
A category has all countable colimits if every functor `J ⥤ C` with a `CountableCategory J`
instance and `J : Type` has a colimit.
-/
class HasCountableColimits : Prop where
  /-- `C` has all limits over any type `J` whose objects and morphisms lie in the same universe
  and which has countably many objects and morphisms -/
  out (J : Type) [SmallCategory J] [CountableCategory J] : HasColimitsOfShape J C


instance (priority := 100) hasFiniteColimits_of_hasCountableColimits [HasCountableColimits C] :
    HasFiniteColimits C where
  out J := HasCountableColimits.out J


instance (priority := 100) hasCountableColimits_of_hasColimits [HasColimits C] :
    HasCountableColimits C where
  out := inferInstance


universe v in
instance [Category.{v} J] [CountableCategory J] [HasCountableColimits C] : HasColimitsOfShape J C :=
  have : HasColimitsOfShape (HomAsType J) C := HasCountableColimits.out (HomAsType J)
  hasColimitsOfShape_of_equivalence (homAsTypeEquiv J)


/-- A category has countable coproducts if it has all coproducts indexed by countable types. -/
class HasCountableCoproducts where
  out (J : Type) [Countable J] : HasCoproductsOfShape J C


instance (priority := 100) hasCountableCoproducts_of_hasCoproducts [HasCoproducts C] :
    HasCountableCoproducts C where
  out _ :=
    have : HasCoproducts.{0} C := has_smallest_coproducts_of_hasCoproducts
    inferInstance


instance [HasCountableCoproducts C] : HasCoproductsOfShape J C :=
  have : Countable (Shrink.{0} J) := Countable.of_equiv _ (equivShrink.{0} J)
  have : HasColimitsOfShape (Discrete (Shrink.{0} J)) C := HasCountableCoproducts.out _
  hasColimitsOfShape_of_equivalence (Discrete.equivalence (equivShrink.{0} J)).symm


instance (priority := 100) hasCountableCoproducts_of_hasCountableColimits [HasCountableColimits C] :
    HasCountableCoproducts C where
  out _ := inferInstance


instance (priority := 100) hasFiniteCoproducts_of_hasCountableCoproducts
    [HasCountableCoproducts C] : HasFiniteCoproducts C where
  out _ := inferInstance


/-- The object part of the initial functor `ℕᵒᵖ ⥤ J` -/
noncomputable def sequentialFunctor_obj : ℕ → J := fun
  | .zero => (exists_surjective_nat _).choose 0
  | .succ n => (IsFilteredOrEmpty.cocone_objs ((exists_surjective_nat _).choose n)
      (sequentialFunctor_obj n)).choose


theorem sequentialFunctor_map : Monotone (sequentialFunctor_obj J) :=
  monotone_nat_of_le_succ fun n ↦
    leOfHom (IsFilteredOrEmpty.cocone_objs ((exists_surjective_nat _).choose n)
      (sequentialFunctor_obj J n)).choose_spec.choose_spec.choose


/--
The initial functor `ℕᵒᵖ ⥤ J`, which allows us to turn cofiltered limits over countable preorders
into sequential limits.
-/
noncomputable def sequentialFunctor : ℕ ⥤ J where
  obj n := sequentialFunctor_obj J n
  map h := homOfLE (sequentialFunctor_map J (leOfHom h))


theorem sequentialFunctor_final_aux (j : J) : ∃ (n : ℕ), j ≤ sequentialFunctor_obj J n := by
  /-
    J : Type u_2
    inst✝² : Countable J
    inst✝¹ : Preorder J
    inst✝ : CategoryTheory.IsFiltered J
    j : J
    ⊢ Exists fun n => LE.le j (CategoryTheory.Limits.IsFiltered.sequentialFunctor_ …
  -/
  obtain ⟨m, h⟩ := (exists_surjective_nat _).choose_spec j
  /-
    case intro
    J : Type u_2
    inst✝² : Countable J
    inst✝¹ : Preorder J
    inst✝ : CategoryTheory.IsFiltered J
    j : J
    m : Nat
    h : Eq (⋯.choose m) j
    ⊢ Exists fun n => LE.le j (CategoryTheory.Limits.IsFiltered.sequentialFunctor_ …
  -/
  refine ⟨m + 1, ?_⟩
  simpa only [h] using leOfHom (IsFilteredOrEmpty.cocone_objs ((exists_surjective_nat _).choose m)
    (sequentialFunctor_obj J m)).choose_spec.choose


instance sequentialFunctor_final : (sequentialFunctor J).Final where
  out d := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.23627, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsFiltered J
      d : J
      ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow d (CategoryTheory …
    -/
    obtain ⟨n, (g : d ≤ (sequentialFunctor J).obj n)⟩ := sequentialFunctor_final_aux J d
    have : Nonempty (StructuredArrow d (sequentialFunctor J)) :=
      ⟨StructuredArrow.mk (homOfLE g)⟩
    /-
      case intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.23627, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsFiltered J
      d : J
      n : Nat
      g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
      this : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFil …
      ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow d (CategoryTheory …
    -/
    apply isConnected_of_zigzag
    /-
      case intro.h
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.23627, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsFiltered J
      d : J
      n : Nat
      g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
      this : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFil …
      ⊢ ∀ (j₁ j₂ : CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFiltere …
    -/
    refine fun i j ↦ ⟨[j], ?_⟩
    simp only [List.chain_cons, Zag, List.Chain.nil, and_true, ne_eq, not_false_eq_true,
      List.getLast_cons, not_true_eq_false, List.getLast_singleton', reduceCtorEq]
    /-
      case intro.h
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.23627, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsFiltered J
      d : J
      n : Nat
      g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
      this : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFil …
      i j : CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFiltered.seque …
      ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
    -/
    clear! C
    /-
      case intro.h
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsFiltered J
      d : J
      n : Nat
      g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
      this : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFil …
      i j : CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFiltered.seque …
      ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
    -/
    wlog h : j.right ≤ i.right
      /-
        case intro.h.inr
        J : Type u_2
        inst✝² : Countable J
        inst✝¹ : Preorder J
        inst✝ : CategoryTheory.IsFiltered J
        d : J
        n : Nat
        g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
        this✝ : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFi …
        i j : CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFiltered.seque …
        this : ∀ (J : Type u_2) [inst : Countable J] [inst_1 : Preorder J] [inst_2 : C …
        h : Not (LE.le j.right i.right)
        ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
      -/
    · exact or_comm.1 (this J d n g inferInstance j i (le_of_lt (not_le.mp h)))
      /-
        🎉 no goals
      -/
      /-
        J✝ : Type u_2
        inst✝⁵ : Countable J✝
        inst✝⁴ : Preorder J✝
        inst✝³ : CategoryTheory.IsFiltered J✝
        J : Type u_2
        inst✝² : Countable J
        inst✝¹ : Preorder J
        inst✝ : CategoryTheory.IsFiltered J
        d : J
        n : Nat
        g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
        this : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFil …
        i j : CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFiltered.seque …
        h : LE.le j.right i.right
        ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
      -/
    · right
      /-
        case h
        J✝ : Type u_2
        inst✝⁵ : Countable J✝
        inst✝⁴ : Preorder J✝
        inst✝³ : CategoryTheory.IsFiltered J✝
        J : Type u_2
        inst✝² : Countable J
        inst✝¹ : Preorder J
        inst✝ : CategoryTheory.IsFiltered J
        d : J
        n : Nat
        g : LE.le d ((CategoryTheory.Limits.IsFiltered.sequentialFunctor J).obj n)
        this : Nonempty (CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFil …
        i j : CategoryTheory.StructuredArrow d (CategoryTheory.Limits.IsFiltered.seque …
        h : LE.le j.right i.right
        ⊢ Nonempty (Quiver.Hom j i)
      -/
      exact ⟨StructuredArrow.homMk (homOfLE h) rfl⟩
      /-
        🎉 no goals
      -/


/-- The object part of the initial functor `ℕᵒᵖ ⥤ J` -/
noncomputable def sequentialFunctor_obj : ℕ → J := fun
  | .zero => (exists_surjective_nat _).choose 0
  | .succ n => (IsCofilteredOrEmpty.cone_objs ((exists_surjective_nat _).choose n)
      (sequentialFunctor_obj n)).choose


theorem sequentialFunctor_map : Antitone (sequentialFunctor_obj J) :=
  antitone_nat_of_succ_le fun n ↦
    leOfHom (IsCofilteredOrEmpty.cone_objs ((exists_surjective_nat _).choose n)
      (sequentialFunctor_obj J n)).choose_spec.choose_spec.choose


/--
The initial functor `ℕᵒᵖ ⥤ J`, which allows us to turn cofiltered limits over countable preorders
into sequential limits.

TODO: redefine this as `(IsFiltered.sequentialFunctor Jᵒᵖ).leftOp`. This would need API for initial/
final functors of the form `leftOp`/`rightOp`.
-/
noncomputable def sequentialFunctor : ℕᵒᵖ ⥤ J where
  obj n := sequentialFunctor_obj J (unop n)
  map h := homOfLE (sequentialFunctor_map J (leOfHom h.unop))


theorem sequentialFunctor_initial_aux (j : J) : ∃ (n : ℕ), sequentialFunctor_obj J n ≤ j := by
  /-
    J : Type u_2
    inst✝² : Countable J
    inst✝¹ : Preorder J
    inst✝ : CategoryTheory.IsCofiltered J
    j : J
    ⊢ Exists fun n => LE.le (CategoryTheory.Limits.IsCofiltered.sequentialFunctor_ …
  -/
  obtain ⟨m, h⟩ := (exists_surjective_nat _).choose_spec j
  /-
    case intro
    J : Type u_2
    inst✝² : Countable J
    inst✝¹ : Preorder J
    inst✝ : CategoryTheory.IsCofiltered J
    j : J
    m : Nat
    h : Eq (⋯.choose m) j
    ⊢ Exists fun n => LE.le (CategoryTheory.Limits.IsCofiltered.sequentialFunctor_ …
  -/
  refine ⟨m + 1, ?_⟩
  simpa only [h] using leOfHom (IsCofilteredOrEmpty.cone_objs ((exists_surjective_nat _).choose m)
    (sequentialFunctor_obj J m)).choose_spec.choose


instance sequentialFunctor_initial : (sequentialFunctor J).Initial where
  out d := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.28832, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsCofiltered J
      d : J
      ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (CategoryTheory …
    -/
    obtain ⟨n, (g : (sequentialFunctor J).obj ⟨n⟩ ≤ d)⟩ := sequentialFunctor_initial_aux J d
    have : Nonempty (CostructuredArrow (sequentialFunctor J) d) :=
      ⟨CostructuredArrow.mk (homOfLE g)⟩
    /-
      case intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.28832, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsCofiltered J
      d : J
      n : Nat
      g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
      this : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCof …
      ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (CategoryTheory …
    -/
    apply isConnected_of_zigzag
    /-
      case intro.h
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.28832, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsCofiltered J
      d : J
      n : Nat
      g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
      this : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCof …
      ⊢ ∀ (j₁ j₂ : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCofilte …
    -/
    refine fun i j ↦ ⟨[j], ?_⟩
    simp only [List.chain_cons, Zag, List.Chain.nil, and_true, ne_eq, not_false_eq_true,
      List.getLast_cons, not_true_eq_false, List.getLast_singleton', reduceCtorEq]
    /-
      case intro.h
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.28832, u_1} C
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsCofiltered J
      d : J
      n : Nat
      g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
      this : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCof …
      i j : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCofiltered.seq …
      ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
    -/
    clear! C
    /-
      case intro.h
      J : Type u_2
      inst✝² : Countable J
      inst✝¹ : Preorder J
      inst✝ : CategoryTheory.IsCofiltered J
      d : J
      n : Nat
      g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
      this : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCof …
      i j : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCofiltered.seq …
      ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
    -/
    wlog h : (unop i.left) ≤ (unop j.left)
      /-
        case intro.h.inr
        J : Type u_2
        inst✝² : Countable J
        inst✝¹ : Preorder J
        inst✝ : CategoryTheory.IsCofiltered J
        d : J
        n : Nat
        g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
        this✝ : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCo …
        i j : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCofiltered.seq …
        this : ∀ (J : Type u_2) [inst : Countable J] [inst_1 : Preorder J] [inst_2 : C …
        h : Not (LE.le (Opposite.unop i.left) (Opposite.unop j.left))
        ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
      -/
    · exact or_comm.1 (this J d n g inferInstance j i (le_of_lt (not_le.mp h)))
      /-
        🎉 no goals
      -/
      /-
        J✝ : Type u_2
        inst✝⁵ : Countable J✝
        inst✝⁴ : Preorder J✝
        inst✝³ : CategoryTheory.IsCofiltered J✝
        J : Type u_2
        inst✝² : Countable J
        inst✝¹ : Preorder J
        inst✝ : CategoryTheory.IsCofiltered J
        d : J
        n : Nat
        g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
        this : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCof …
        i j : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCofiltered.seq …
        h : LE.le (Opposite.unop i.left) (Opposite.unop j.left)
        ⊢ Or (Nonempty (Quiver.Hom i j)) (Nonempty (Quiver.Hom j i))
      -/
    · right
      /-
        case h
        J✝ : Type u_2
        inst✝⁵ : Countable J✝
        inst✝⁴ : Preorder J✝
        inst✝³ : CategoryTheory.IsCofiltered J✝
        J : Type u_2
        inst✝² : Countable J
        inst✝¹ : Preorder J
        inst✝ : CategoryTheory.IsCofiltered J
        d : J
        n : Nat
        g : LE.le ((CategoryTheory.Limits.IsCofiltered.sequentialFunctor J).obj { unop …
        this : Nonempty (CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCof …
        i j : CategoryTheory.CostructuredArrow (CategoryTheory.Limits.IsCofiltered.seq …
        h : LE.le (Opposite.unop i.left) (Opposite.unop j.left)
        ⊢ Nonempty (Quiver.Hom j i)
      -/
      exact ⟨CostructuredArrow.homMk (homOfLE h).op rfl⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-01")] alias sequentialFunctor := IsCofiltered.sequentialFunctor

@[deprecated (since := "2024-11-01")] alias sequentialFunctor_obj :=
  IsCofiltered.sequentialFunctor_obj

@[deprecated (since := "2024-11-01")] alias sequentialFunctor_map :=
  IsCofiltered.sequentialFunctor_map

@[deprecated (since := "2024-11-01")] alias sequentialFunctor_initial_aux :=
  IsCofiltered.sequentialFunctor_initial_aux

@[deprecated (since := "2024-11-01")] alias sequentialFunctor_initial :=
  IsCofiltered.sequentialFunctor_initial

