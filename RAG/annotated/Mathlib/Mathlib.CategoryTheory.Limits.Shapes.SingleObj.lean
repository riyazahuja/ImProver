/-- The induced `G`-action on the target of `J : SingleObj G ⥤ Type u`. -/
instance (J : SingleObj M ⥤ Type u) : MulAction M (J.obj (SingleObj.star M)) where
  smul g x := J.map g x
  one_smul x := by
    /-
      M G : Type v
      inst✝¹ : Monoid M
      inst✝ : Group G
      J : CategoryTheory.Functor (CategoryTheory.SingleObj M) (Type u)
      x : J.obj (CategoryTheory.SingleObj.star M)
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    show J.map (𝟙 _) x = x
    /-
      M G : Type v
      inst✝¹ : Monoid M
      inst✝ : Group G
      J : CategoryTheory.Functor (CategoryTheory.SingleObj M) (Type u)
      x : J.obj (CategoryTheory.SingleObj.star M)
      ⊢ Eq (J.map (CategoryTheory.CategoryStruct.id (CategoryTheory.SingleObj.star M …
    -/
    simp only [FunctorToTypes.map_id_apply]
    /-
      🎉 no goals
    -/
  mul_smul g h x := by
    /-
      M G : Type v
      inst✝¹ : Monoid M
      inst✝ : Group G
      J : CategoryTheory.Functor (CategoryTheory.SingleObj M) (Type u)
      g h : M
      x : J.obj (CategoryTheory.SingleObj.star M)
      ⊢ Eq (HSMul.hSMul (HMul.hMul g h) x) (HSMul.hSMul g (HSMul.hSMul h x))
    -/
    show J.map (g * h) x = (J.map h ≫ J.map g) x
    /-
      M G : Type v
      inst✝¹ : Monoid M
      inst✝ : Group G
      J : CategoryTheory.Functor (CategoryTheory.SingleObj M) (Type u)
      g h : M
      x : J.obj (CategoryTheory.SingleObj.star M)
      ⊢ Eq (J.map (HMul.hMul g h) x) (CategoryTheory.CategoryStruct.comp (J.map h) ( …
    -/
    rw [← SingleObj.comp_as_mul]
      /-
        M G : Type v
        inst✝¹ : Monoid M
        inst✝ : Group G
        J : CategoryTheory.Functor (CategoryTheory.SingleObj M) (Type u)
        g h : M
        x : J.obj (CategoryTheory.SingleObj.star M)
        ⊢ Eq (J.map (CategoryTheory.CategoryStruct.comp h g) x) (CategoryTheory.Catego …
      -/
    · simp only [FunctorToTypes.map_comp_apply, types_comp_apply]
      /-
        M G : Type v
        inst✝¹ : Monoid M
        inst✝ : Group G
        J : CategoryTheory.Functor (CategoryTheory.SingleObj M) (Type u)
        g h : M
        x : J.obj (CategoryTheory.SingleObj.star M)
        ⊢ Eq (J.map g (J.map h x)) (J.map g (J.map h x))
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- The equivalence between sections of `J : SingleObj M ⥤ Type u` and fixed points of the
induced action on `J.obj (SingleObj.star M)`. -/
@[simps]
def Types.sections.equivFixedPoints :
    J.sections ≃ MulAction.fixedPoints M (J.obj (SingleObj.star M)) where
  toFun s := ⟨s.val _, s.property⟩
  invFun p := ⟨fun _ ↦ p.val, p.property⟩
  left_inv _ := rfl
  right_inv _ := rfl


/-- The limit of `J : SingleObj M ⥤ Type u` is equivalent to the fixed points of the
induced action on `J.obj (SingleObj.star M)`. -/
@[simps!]
noncomputable def Types.limitEquivFixedPoints :
    limit J ≃ MulAction.fixedPoints M (J.obj (SingleObj.star M)) :=
  (Types.limitEquivSections J).trans (Types.sections.equivFixedPoints J)


/-- The relation used to construct colimits in types for `J : SingleObj G ⥤ Type u` is
equivalent to the `MulAction.orbitRel` equivalence relation on `J.obj (SingleObj.star G)`. -/
lemma Types.Quot.Rel.iff_orbitRel (x y : J.obj (SingleObj.star G)) :
    Types.Quot.Rel J ⟨SingleObj.star G, x⟩ ⟨SingleObj.star G, y⟩
    ↔ MulAction.orbitRel G (J.obj (SingleObj.star G)) x y := by
  /-
    G : Type v
    inst✝ : Group G
    J : CategoryTheory.Functor (CategoryTheory.SingleObj G) (Type u)
    x y : J.obj (CategoryTheory.SingleObj.star G)
    ⊢ Iff (CategoryTheory.Limits.Types.Quot.Rel J ⟨CategoryTheory.SingleObj.star G …
  -/
  have h (g : G) : y = g • x ↔ g • x = y := ⟨symm, symm⟩
  /-
    G : Type v
    inst✝ : Group G
    J : CategoryTheory.Functor (CategoryTheory.SingleObj G) (Type u)
    x y : J.obj (CategoryTheory.SingleObj.star G)
    h : ∀ (g : G), Iff (Eq y (HSMul.hSMul g x)) (Eq (HSMul.hSMul g x) y)
    ⊢ Iff (CategoryTheory.Limits.Types.Quot.Rel J ⟨CategoryTheory.SingleObj.star G …
  -/
  conv => rhs; rw [Setoid.comm']
  /-
    G : Type v
    inst✝ : Group G
    J : CategoryTheory.Functor (CategoryTheory.SingleObj G) (Type u)
    x y : J.obj (CategoryTheory.SingleObj.star G)
    h : ∀ (g : G), Iff (Eq y (HSMul.hSMul g x)) (Eq (HSMul.hSMul g x) y)
    ⊢ Iff (CategoryTheory.Limits.Types.Quot.Rel J ⟨CategoryTheory.SingleObj.star G …
  -/
  show (∃ g : G, y = g • x) ↔ (∃ g : G, g • x = y)
  /-
    G : Type v
    inst✝ : Group G
    J : CategoryTheory.Functor (CategoryTheory.SingleObj G) (Type u)
    x y : J.obj (CategoryTheory.SingleObj.star G)
    h : ∀ (g : G), Iff (Eq y (HSMul.hSMul g x)) (Eq (HSMul.hSMul g x) y)
    ⊢ Iff (Exists fun g => Eq y (HSMul.hSMul g x)) (Exists fun g => Eq (HSMul.hSMu …
  -/
  conv => lhs; simp only [h]
  /-
    🎉 no goals
  -/


/-- The explicit quotient construction of the colimit of `J : SingleObj G ⥤ Type u` is
equivalent to the quotient of `J.obj (SingleObj.star G)` by the induced action. -/
@[simps]
def Types.Quot.equivOrbitRelQuotient :
    Types.Quot J ≃ MulAction.orbitRel.Quotient G (J.obj (SingleObj.star G)) where
  toFun := Quot.lift (fun p => ⟦p.2⟧) <| fun a b h => Quotient.sound <|
    (Types.Quot.Rel.iff_orbitRel J a.2 b.2).mp h
  invFun := Quot.lift (fun x => Quot.mk _ ⟨SingleObj.star G, x⟩) <| fun a b h =>
    Quot.sound <| (Types.Quot.Rel.iff_orbitRel J a b).mpr h
  left_inv := fun x => Quot.inductionOn x (fun _ ↦ rfl)
  right_inv := fun x => Quot.inductionOn x (fun _ ↦ rfl)


/-- The colimit of `J : SingleObj G ⥤ Type u` is equivalent to the quotient of
`J.obj (SingleObj.star G)` by the induced action. -/
@[simps!]
noncomputable def Types.colimitEquivQuotient :
    colimit J ≃ MulAction.orbitRel.Quotient G (J.obj (SingleObj.star G)) :=
  (Types.colimitEquivQuot J).trans (Types.Quot.equivOrbitRelQuotient J)


