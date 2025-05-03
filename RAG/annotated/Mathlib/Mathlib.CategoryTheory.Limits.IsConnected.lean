/-- The functor mapping every object to `PUnit`. -/
def constPUnitFunctor : C ⥤ Type w := (Functor.const C).obj PUnit.{w + 1}


/-- The cocone on `constPUnitFunctor` with cone point `PUnit`. -/
@[simps]
def pUnitCocone : Cocone (constPUnitFunctor.{w} C) where
  pt := PUnit
  ι := { app := fun _ => id }


/-- If `C` is connected, the cocone on `constPUnitFunctor` with cone point `PUnit` is a colimit
    cocone. -/
noncomputable def isColimitPUnitCocone [IsConnected C] : IsColimit (pUnitCocone.{w} C) where
  desc s := s.ι.app Classical.ofNonempty
  fac s j := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.IsConnected C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.Types.constPUnitFuncto …
      j : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Types.pUnitCo …
    -/
    ext ⟨⟩
    /-
      case h.unit
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.IsConnected C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.Types.constPUnitFuncto …
      j : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Types.pUnitCo …
    -/
    apply constant_of_preserves_morphisms (s.ι.app · PUnit.unit)
    /-
      case h.unit.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.IsConnected C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.Types.constPUnitFuncto …
      j : C
      ⊢ ∀ (j₁ j₂ : C), Quiver.Hom j₁ j₂ → Eq (s.ι.app j₁ PUnit.unit) (s.ι.app j₂ PUn …
    -/
    intros X Y f
    /-
      case h.unit.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.IsConnected C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.Types.constPUnitFuncto …
      j X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (s.ι.app X PUnit.unit) (s.ι.app Y PUnit.unit)
    -/
    exact congrFun (s.ι.naturality f).symm PUnit.unit
    /-
      🎉 no goals
    -/
  uniq s m h := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.IsConnected C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.Types.constPUnitFuncto …
      m : Quiver.Hom (CategoryTheory.Limits.Types.pUnitCocone C).pt s.pt
      h : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq m ((fun s => s.ι.app Classical.ofNonempty) s)
    -/
    ext ⟨⟩
    /-
      case h.unit
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.IsConnected C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.Types.constPUnitFuncto …
      m : Quiver.Hom (CategoryTheory.Limits.Types.pUnitCocone C).pt s.pt
      h : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq (m PUnit.unit) ((fun s => s.ι.app Classical.ofNonempty) s PUnit.unit)
    -/
    simp [← h Classical.ofNonempty]
    /-
      🎉 no goals
    -/


instance instHasColimitConstPUnitFunctor [IsConnected C] : HasColimit (constPUnitFunctor.{w} C) :=
  ⟨_, isColimitPUnitCocone _⟩


instance instSubsingletonColimitPUnit
    [IsPreconnected C] [HasColimit (constPUnitFunctor.{w} C)] :
    Subsingleton (colimit (constPUnitFunctor.{w} C)) where
  allEq a b := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.IsPreconnected C
      inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
      a b : CategoryTheory.Limits.colimit (CategoryTheory.Limits.Types.constPUnitFun …
      ⊢ Eq a b
    -/
    obtain ⟨c, ⟨⟩, rfl⟩ := jointly_surjective' a
    /-
      case intro.intro.unit
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.IsPreconnected C
      inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
      b : CategoryTheory.Limits.colimit (CategoryTheory.Limits.Types.constPUnitFunct …
      c : C
      ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.Types.constPUnitF …
    -/
    obtain ⟨d, ⟨⟩, rfl⟩ := jointly_surjective' b
    /-
      case intro.intro.unit.intro.intro.unit
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.IsPreconnected C
      inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
      c d : C
      ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.Types.constPUnitF …
    -/
    apply constant_of_preserves_morphisms (colimit.ι (constPUnitFunctor C) · PUnit.unit)
    /-
      case intro.intro.unit.intro.intro.unit.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.IsPreconnected C
      inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
      c d : C
      ⊢ ∀ (j₁ j₂ : C), Quiver.Hom j₁ j₂ → Eq (CategoryTheory.Limits.colimit.ι (Categ …
    -/
    exact fun c d f => colimit_sound f rfl
    /-
      🎉 no goals
    -/


/-- Given a connected index category, the colimit of the constant unit-valued functor is `PUnit`. -/
noncomputable def colimitConstPUnitIsoPUnit [IsConnected C] :
    colimit (constPUnitFunctor.{w} C) ≅ PUnit.{w + 1} :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit _) (isColimitPUnitCocone.{w} C)


/-- Let `F` be a `Type`-valued functor. If two elements `a : F c` and `b : F d` represent the same
element of `colimit F`, then `c` and `d` are related by a `Zigzag`. -/
theorem zigzag_of_eqvGen_quot_rel (F : C ⥤ Type w) (c d : Σ j, F.obj j)
    (h : Relation.EqvGen (Quot.Rel F) c d) : Zigzag c.1 d.1 := by
  induction h with
  | rel _ _ h => exact Zigzag.of_hom <| Exists.choose h
  | refl _ => exact Zigzag.refl _
  | symm _ _ _ ih => exact zigzag_symmetric ih
  | trans _ _ _ _ _ ih₁ ih₂ => exact ih₁.trans ih₂


/-- An index category is connected iff the colimit of the constant singleton-valued functor is a
singleton. -/
theorem isConnected_iff_colimit_constPUnitFunctor_iso_pUnit
    [HasColimit (constPUnitFunctor.{w} C)] :
    IsConnected C ↔ Nonempty (colimit (constPUnitFunctor.{w} C) ≅ PUnit) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
    ⊢ Iff (CategoryTheory.IsConnected C) (Nonempty (CategoryTheory.Iso (CategoryTh …
  -/
  refine ⟨fun _ => ⟨colimitConstPUnitIsoPUnit.{w} C⟩, fun ⟨h⟩ => ?_⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
    x✝ : Nonempty (CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheo …
    h : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.T …
    ⊢ CategoryTheory.IsConnected C
  -/
  have : Nonempty C := nonempty_of_nonempty_colimit <| Nonempty.map h.inv inferInstance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
    x✝ : Nonempty (CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheo …
    h : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.T …
    this : Nonempty C
    ⊢ CategoryTheory.IsConnected C
  -/
  refine zigzag_isConnected <| fun c d => ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
    x✝ : Nonempty (CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheo …
    h : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.T …
    this : Nonempty C
    c d : C
    ⊢ CategoryTheory.Zigzag c d
  -/
  refine zigzag_of_eqvGen_quot_rel _ (constPUnitFunctor C) ⟨c, PUnit.unit⟩ ⟨d, PUnit.unit⟩ ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUn …
    x✝ : Nonempty (CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheo …
    h : CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.Limits.T …
    this : Nonempty C
    c d : C
    ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel (CategoryTheory.Limits …
  -/
  exact colimit_eq <| h.toEquiv.injective rfl
  /-
    🎉 no goals
  -/


theorem isConnected_iff_isColimit_pUnitCocone :
    IsConnected C ↔ Nonempty (IsColimit (pUnitCocone.{w} C)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (CategoryTheory.IsConnected C) (Nonempty (CategoryTheory.Limits.IsColimi …
  -/
  refine ⟨fun inst => ⟨isColimitPUnitCocone C⟩, fun ⟨h⟩ => ?_⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    x✝ : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pU …
    h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pUnitCocone C)
    ⊢ CategoryTheory.IsConnected C
  -/
  let colimitCocone : ColimitCocone (constPUnitFunctor C) := ⟨pUnitCocone.{w} C, h⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    x✝ : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pU …
    h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pUnitCocone C)
    colimitCocone : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.Typ …
    ⊢ CategoryTheory.IsConnected C
  -/
  have : HasColimit (constPUnitFunctor.{w} C) := ⟨⟨colimitCocone⟩⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    x✝ : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pU …
    h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pUnitCocone C)
    colimitCocone : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.Typ …
    this : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUni …
    ⊢ CategoryTheory.IsConnected C
  -/
  simp only [isConnected_iff_colimit_constPUnitFunctor_iso_pUnit.{w} C]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    x✝ : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pU …
    h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.pUnitCocone C)
    colimitCocone : CategoryTheory.Limits.ColimitCocone (CategoryTheory.Limits.Typ …
    this : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.Types.constPUni …
    ⊢ Nonempty (CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory. …
  -/
  exact ⟨colimit.isoColimitCocone colimitCocone⟩
  /-
    🎉 no goals
  -/


/-- The domain of a final functor is connected if and only if its codomain is connected. -/
theorem isConnected_iff_of_final (F : C ⥤ D) [F.Final] : IsConnected C ↔ IsConnected D := by
  rw [isConnected_iff_colimit_constPUnitFunctor_iso_pUnit.{max v u v₂ u₂} C,
    isConnected_iff_colimit_constPUnitFunctor_iso_pUnit.{max v u v₂ u₂} D]
  exact Equiv.nonempty_congr <| Iso.isoCongrLeft <|
    CategoryTheory.Functor.Final.colimitIso F <| constPUnitFunctor.{max u v u₂ v₂} D


/-- The domain of an initial functor is connected if and only if its codomain is connected. -/
theorem isConnected_iff_of_initial (F : C ⥤ D) [F.Initial] : IsConnected C ↔ IsConnected D := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.Initial
    ⊢ Iff (CategoryTheory.IsConnected C) (CategoryTheory.IsConnected D)
  -/
  rw [← isConnected_op_iff_isConnected C, ← isConnected_op_iff_isConnected D]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.Initial
    ⊢ Iff (CategoryTheory.IsConnected (Opposite C)) (CategoryTheory.IsConnected (O …
  -/
  exact isConnected_iff_of_final F.op
  /-
    🎉 no goals
  -/


