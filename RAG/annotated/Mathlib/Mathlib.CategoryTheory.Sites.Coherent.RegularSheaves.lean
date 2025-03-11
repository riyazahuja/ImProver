/-- A presieve is *regular* if it consists of a single effective epimorphism. -/
class Presieve.regular {X : C} (R : Presieve X) : Prop where
  /-- `R` consists of a single epimorphism. -/
  single_epi : ∃ (Y : C) (f : Y ⟶ X), R = Presieve.ofArrows (fun (_ : Unit) ↦ Y)
    (fun (_ : Unit) ↦ f) ∧ EffectiveEpi f


lemma equalizerCondition_w (P : Cᵒᵖ ⥤ D) {X B : C} {π : X ⟶ B} (c : PullbackCone π π) :
    P.map π.op ≫ P.map c.fst.op = P.map π.op ≫ P.map c.snd.op := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_5, u_2} D
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map π.op) (P.map c.fst.op)) (Categ …
  -/
  simp only [← Functor.map_comp, ← op_comp, c.condition]
  /-
    🎉 no goals
  -/


/--
A contravariant functor on `C` satisfies `SingleEqualizerCondition` with respect to a morphism `π`
if it takes its kernel pair to an equalizer diagram.
-/
def SingleEqualizerCondition (P : Cᵒᵖ ⥤ D) ⦃X B : C⦄ (π : X ⟶ B) : Prop :=
  ∀ (c : PullbackCone π π) (_ : IsLimit c),
    Nonempty (IsLimit (Fork.ofι (P.map π.op) (equalizerCondition_w P c)))


/--
A contravariant functor on `C` satisfies `EqualizerCondition` if it takes kernel pairs of effective
epimorphisms to equalizer diagrams.
-/
def EqualizerCondition (P : Cᵒᵖ ⥤ D) : Prop :=
  ∀ ⦃X B : C⦄ (π : X ⟶ B) [EffectiveEpi π], SingleEqualizerCondition P π


/-- The equalizer condition is preserved by natural isomorphism. -/
theorem equalizerCondition_of_natIso {P P' : Cᵒᵖ ⥤ D} (i : P ≅ P')
    (hP : EqualizerCondition P) : EqualizerCondition P' := fun X B π _ c hc ↦
   /-
     C : Type u_1
     D : Type u_2
     inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
     inst✝ : CategoryTheory.Category.{u_5, u_2} D
     P P' : CategoryTheory.Functor (Opposite C) D
     i : CategoryTheory.Iso P P'
     hP : CategoryTheory.regularTopology.EqualizerCondition P
     X B : C
     π : Quiver.Hom X B
     x✝ : CategoryTheory.EffectiveEpi π
     c : CategoryTheory.Limits.PullbackCone π π
     hc : CategoryTheory.Limits.IsLimit c
     ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.app { unop := X }).hom (P'.map c.f …
   -/
   /-
     🎉 no goals
   -/
   /-
     🎉 no goals
   -/
  ⟨Fork.isLimitOfIsos _ (hP π c hc).some _ (i.app _) (i.app _) (i.app _)⟩
   /-
     🎉 no goals
   -/


/-- Precomposing with a pullback-preserving functor preserves the equalizer condition. -/
theorem equalizerCondition_precomp_of_preservesPullback (P : Cᵒᵖ ⥤ D) (F : E ⥤ C)
    [∀ {X B} (π : X ⟶ B) [EffectiveEpi π], PreservesLimit (cospan π π) F]
    [F.PreservesEffectiveEpis] (hP : EqualizerCondition P) : EqualizerCondition (F.op ⋙ P) := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Category.{u_6, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    F : CategoryTheory.Functor E C
    inst✝¹ : ∀ {X B : E} (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi  …
    inst✝ : F.PreservesEffectiveEpis
    hP : CategoryTheory.regularTopology.EqualizerCondition P
    ⊢ CategoryTheory.regularTopology.EqualizerCondition (F.op.comp P)
  -/
  intro X B π _ c hc
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    inst✝³ : CategoryTheory.Category.{u_6, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    F : CategoryTheory.Functor E C
    inst✝² : ∀ {X B : E} (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi  …
    inst✝¹ : F.PreservesEffectiveEpis
    hP : CategoryTheory.regularTopology.EqualizerCondition P
    X B : E
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι ((F. …
  -/
  have h : P.map (F.map π).op = (F.op ⋙ P).map π.op := by simp
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    inst✝³ : CategoryTheory.Category.{u_6, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    F : CategoryTheory.Functor E C
    inst✝² : ∀ {X B : E} (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi  …
    inst✝¹ : F.PreservesEffectiveEpis
    hP : CategoryTheory.regularTopology.EqualizerCondition P
    X B : E
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    h : Eq (P.map (F.map π).op) ((F.op.comp P).map π.op)
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι ((F. …
  -/
  refine ⟨(IsLimit.equivIsoLimit (ForkOfι.ext ?_ _ h)) ?_⟩
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} E
      P : CategoryTheory.Functor (Opposite C) D
      F : CategoryTheory.Functor E C
      inst✝² : ∀ {X B : E} (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi  …
      inst✝¹ : F.PreservesEffectiveEpis
      hP : CategoryTheory.regularTopology.EqualizerCondition P
      X B : E
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.EffectiveEpi π
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      h : Eq (P.map (F.map π).op) ((F.op.comp P).map π.op)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map (F.map π).op) ((F.op.comp P).m …
    -/
  · simp only [Functor.comp_map, op_map, Quiver.Hom.unop_op, ← map_comp, ← op_comp, c.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
      inst✝³ : CategoryTheory.Category.{u_6, u_3} E
      P : CategoryTheory.Functor (Opposite C) D
      F : CategoryTheory.Functor E C
      inst✝² : ∀ {X B : E} (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi  …
      inst✝¹ : F.PreservesEffectiveEpis
      hP : CategoryTheory.regularTopology.EqualizerCondition P
      X B : E
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.EffectiveEpi π
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      h : Eq (P.map (F.map π).op) ((F.op.comp P).map π.op)
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map (F.map  …
    -/
  · refine (hP (F.map π) (PullbackCone.mk (F.map c.fst) (F.map c.snd) ?_) ?_).some
      /-
        case refine_2.refine_1
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
        inst✝³ : CategoryTheory.Category.{u_6, u_3} E
        P : CategoryTheory.Functor (Opposite C) D
        F : CategoryTheory.Functor E C
        inst✝² : ∀ {X B : E} (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi  …
        inst✝¹ : F.PreservesEffectiveEpis
        hP : CategoryTheory.regularTopology.EqualizerCondition P
        X B : E
        π : Quiver.Hom X B
        inst✝ : CategoryTheory.EffectiveEpi π
        c : CategoryTheory.Limits.PullbackCone π π
        hc : CategoryTheory.Limits.IsLimit c
        h : Eq (P.map (F.map π).op) ((F.op.comp P).map π.op)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map c.fst) (F.map π)) (CategoryThe …
      -/
    · simp only [← map_comp, c.condition]
      /-
        🎉 no goals
      -/
    · exact (isLimitMapConePullbackConeEquiv F c.condition)
        (isLimitOfPreserves F (hc.ofIsoLimit (PullbackCone.ext (Iso.refl _) (by simp) (by simp))))


/-- The canonical map to the explicit equalizer. -/
def MapToEqualizer (P : Cᵒᵖ ⥤ Type*) {W X B : C} (f : X ⟶ B)
    (g₁ g₂ : W ⟶ X) (w : g₁ ≫ f = g₂ ≫ f) :
    P.obj (op B) → { x : P.obj (op X) | P.map g₁.op x = P.map g₂.op x } := fun t ↦
                    /-
                      C : Type u_1
                      D : Type u_2
                      E : Type u_3
                      inst✝² : CategoryTheory.Category.{?u.15455, u_1} C
                      inst✝¹ : CategoryTheory.Category.{?u.15459, u_2} D
                      inst✝ : CategoryTheory.Category.{?u.15463, u_3} E
                      P : CategoryTheory.Functor (Opposite C) (Type u_4)
                      W X B : C
                      f : Quiver.Hom X B
                      g₁ g₂ : Quiver.Hom W X
                      w : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStruc …
                      t : P.obj { unop := B }
                      ⊢ Membership.mem (setOf fun x => Eq (P.map g₁.op x) (P.map g₂.op x)) (P.map f. …
                    -/
  ⟨P.map f.op t, by simp only [Set.mem_setOf_eq, ← FunctorToTypes.map_comp_apply, ← op_comp, w]⟩
                    /-
                      🎉 no goals
                    -/


theorem EqualizerCondition.bijective_mapToEqualizer_pullback (P : Cᵒᵖ ⥤ Type*)
    (hP : EqualizerCondition P) : ∀ (X B : C) (π : X ⟶ B) [EffectiveEpi π] [HasPullback π π],
    Function.Bijective
      (MapToEqualizer P π (pullback.fst π π) (pullback.snd π π) pullback.condition) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    hP : CategoryTheory.regularTopology.EqualizerCondition P
    ⊢ ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ins …
  -/
  intro X B π _ _
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    hP : CategoryTheory.regularTopology.EqualizerCondition P
    X B : C
    π : Quiver.Hom X B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Categ …
  -/
  specialize hP π _ (pullbackIsPullback π π)
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    hP : Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι ( …
    ⊢ Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Categ …
  -/
  rw [Types.type_equalizer_iff_unique] at hP
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (y : P.obj { unop := X }), Eq (P.map (CategoryTheory.Limits.PullbackCon …
    ⊢ Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Categ …
  -/
  rw [Function.bijective_iff_existsUnique]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (y : P.obj { unop := X }), Eq (P.map (CategoryTheory.Limits.PullbackCon …
    ⊢ ∀ (b : ↑(setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback.fst π π).o …
  -/
  intro ⟨b, hb⟩
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (y : P.obj { unop := X }), Eq (P.map (CategoryTheory.Limits.PullbackCon …
    b : P.obj { unop := X }
    hb : Membership.mem (setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback. …
    ⊢ ExistsUnique fun a => Eq (CategoryTheory.regularTopology.MapToEqualizer P π  …
  -/
  obtain ⟨a, ha₁, ha₂⟩ := hP b hb
  /-
    case intro.intro
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (y : P.obj { unop := X }), Eq (P.map (CategoryTheory.Limits.PullbackCon …
    b : P.obj { unop := X }
    hb : Membership.mem (setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback. …
    a : P.obj { unop := B }
    ha₁ : Eq (P.map π.op a) b
    ha₂ : ∀ (y : P.obj { unop := B }), (fun x => Eq (P.map π.op x) b) y → Eq y a
    ⊢ ExistsUnique fun a => Eq (CategoryTheory.regularTopology.MapToEqualizer P π  …
  -/
  refine ⟨a, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      X B : C
      π : Quiver.Hom X B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      hP : ∀ (y : P.obj { unop := X }), Eq (P.map (CategoryTheory.Limits.PullbackCon …
      b : P.obj { unop := X }
      hb : Membership.mem (setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback. …
      a : P.obj { unop := B }
      ha₁ : Eq (P.map π.op a) b
      ha₂ : ∀ (y : P.obj { unop := B }), (fun x => Eq (P.map π.op x) b) y → Eq y a
      ⊢ (fun a => Eq (CategoryTheory.regularTopology.MapToEqualizer P π (CategoryThe …
    -/
  · simpa [MapToEqualizer] using ha₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      X B : C
      π : Quiver.Hom X B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      hP : ∀ (y : P.obj { unop := X }), Eq (P.map (CategoryTheory.Limits.PullbackCon …
      b : P.obj { unop := X }
      hb : Membership.mem (setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback. …
      a : P.obj { unop := B }
      ha₁ : Eq (P.map π.op a) b
      ha₂ : ∀ (y : P.obj { unop := B }), (fun x => Eq (P.map π.op x) b) y → Eq y a
      ⊢ ∀ (y : P.obj { unop := B }), (fun a => Eq (CategoryTheory.regularTopology.Ma …
    -/
  · simpa [MapToEqualizer] using ha₂
    /-
      🎉 no goals
    -/


theorem EqualizerCondition.mk (P : Cᵒᵖ ⥤ Type*)
    (hP : ∀ (X B : C) (π : X ⟶ B) [EffectiveEpi π] [HasPullback π π], Function.Bijective
    (MapToEqualizer P π (pullback.fst π π) (pullback.snd π π)
    pullback.condition)) : EqualizerCondition P := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
    ⊢ CategoryTheory.regularTopology.EqualizerCondition P
  -/
  intro X B π _ c hc
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.m …
  -/
  have : HasPullback π π := ⟨c, hc⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    this : CategoryTheory.Limits.HasPullback π π
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.m …
  -/
  specialize hP X B π
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    this : CategoryTheory.Limits.HasPullback π π
    hP : Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Ca …
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.m …
  -/
  rw [Types.type_equalizer_iff_unique]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    this : CategoryTheory.Limits.HasPullback π π
    hP : Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Ca …
    ⊢ ∀ (y : P.obj { unop := X }), Eq (P.map c.fst.op y) (P.map c.snd.op y) → Exis …
  -/
  rw [Function.bijective_iff_existsUnique] at hP
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    this : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (b : ↑(setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback.fst π π …
    ⊢ ∀ (y : P.obj { unop := X }), Eq (P.map c.fst.op y) (P.map c.snd.op y) → Exis …
  -/
  intro b hb
  have h₁ : ((pullbackIsPullback π π).conePointUniqueUpToIso hc).hom ≫ c.fst =
    pullback.fst π π := by simp
  have hb' : P.map (pullback.fst π π).op b = P.map (pullback.snd _ _).op b := by
    rw [← h₁, op_comp, FunctorToTypes.map_comp_apply, hb]
    simp [← FunctorToTypes.map_comp_apply, ← op_comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    this : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (b : ↑(setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback.fst π π …
    b : P.obj { unop := X }
    hb : Eq (P.map c.fst.op b) (P.map c.snd.op b)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackIs …
    hb' : Eq (P.map (CategoryTheory.Limits.pullback.fst π π).op b) (P.map (Categor …
    ⊢ ExistsUnique fun x => Eq (P.map π.op x) b
  -/
  obtain ⟨a, ha₁, ha₂⟩ := hP ⟨b, hb'⟩
  /-
    case intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    this : CategoryTheory.Limits.HasPullback π π
    hP : ∀ (b : ↑(setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback.fst π π …
    b : P.obj { unop := X }
    hb : Eq (P.map c.fst.op b) (P.map c.snd.op b)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackIs …
    hb' : Eq (P.map (CategoryTheory.Limits.pullback.fst π π).op b) (P.map (Categor …
    a : P.obj { unop := B }
    ha₁ : Eq (CategoryTheory.regularTopology.MapToEqualizer P π (CategoryTheory.Li …
    ha₂ : ∀ (y : P.obj { unop := B }), (fun a => Eq (CategoryTheory.regularTopolog …
    ⊢ ExistsUnique fun x => Eq (P.map π.op x) b
  -/
  refine ⟨a, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      X B : C
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.EffectiveEpi π
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.Limits.HasPullback π π
      hP : ∀ (b : ↑(setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback.fst π π …
      b : P.obj { unop := X }
      hb : Eq (P.map c.fst.op b) (P.map c.snd.op b)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackIs …
      hb' : Eq (P.map (CategoryTheory.Limits.pullback.fst π π).op b) (P.map (Categor …
      a : P.obj { unop := B }
      ha₁ : Eq (CategoryTheory.regularTopology.MapToEqualizer P π (CategoryTheory.Li …
      ha₂ : ∀ (y : P.obj { unop := B }), (fun a => Eq (CategoryTheory.regularTopolog …
      ⊢ (fun x => Eq (P.map π.op x) b) a
    -/
  · simpa [MapToEqualizer] using ha₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      X B : C
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.EffectiveEpi π
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.Limits.HasPullback π π
      hP : ∀ (b : ↑(setOf fun x => Eq (P.map (CategoryTheory.Limits.pullback.fst π π …
      b : P.obj { unop := X }
      hb : Eq (P.map c.fst.op b) (P.map c.snd.op b)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackIs …
      hb' : Eq (P.map (CategoryTheory.Limits.pullback.fst π π).op b) (P.map (Categor …
      a : P.obj { unop := B }
      ha₁ : Eq (CategoryTheory.regularTopology.MapToEqualizer P π (CategoryTheory.Li …
      ha₂ : ∀ (y : P.obj { unop := B }), (fun a => Eq (CategoryTheory.regularTopolog …
      ⊢ ∀ (y : P.obj { unop := B }), (fun x => Eq (P.map π.op x) b) y → Eq y a
    -/
  · simpa [MapToEqualizer] using ha₂
    /-
      🎉 no goals
    -/


lemma equalizerCondition_w' (P : Cᵒᵖ ⥤ Type*) {X B : C} (π : X ⟶ B)
    [HasPullback π π] : P.map π.op ≫ P.map (pullback.fst π π).op =
    P.map π.op ≫ P.map (pullback.snd π π).op := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map π.op) (P.map (CategoryTheory.L …
  -/
  simp only [← Functor.map_comp, ← op_comp, pullback.condition]
  /-
    🎉 no goals
  -/


lemma mapToEqualizer_eq_comp (P : Cᵒᵖ ⥤ Type*) {X B : C} (π : X ⟶ B) [HasPullback π π] :
    MapToEqualizer P π (pullback.fst π π) (pullback.snd π π) pullback.condition =
    equalizer.lift (P.map π.op) (equalizerCondition_w' P π) ≫
    (Types.equalizerIso _ _).hom := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Eq (CategoryTheory.regularTopology.MapToEqualizer P π (CategoryTheory.Limits …
  -/
  rw [← Iso.comp_inv_eq (α := Types.equalizerIso _ _)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.regularTopology.MapTo …
  -/
  apply equalizer.hom_ext
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- An alternative phrasing of the explicit equalizer condition, using more categorical language. -/
theorem equalizerCondition_iff_isIso_lift (P : Cᵒᵖ ⥤ Type*) : EqualizerCondition P ↔
    ∀ (X B : C) (π : X ⟶ B) [EffectiveEpi π] [HasPullback π π],
      IsIso (equalizer.lift (P.map π.op) (equalizerCondition_w' P π)) := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_5, u_1} C
    P : CategoryTheory.Functor (Opposite C) (Type u_4)
    ⊢ Iff (CategoryTheory.regularTopology.EqualizerCondition P) (∀ (X B : C) (π :  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      ⊢ CategoryTheory.regularTopology.EqualizerCondition P → ∀ (X B : C) (π : Quive …
    -/
  · intro hP X B π _ _
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      hP : CategoryTheory.regularTopology.EqualizerCondition P
      X B : C
      π : Quiver.Hom X B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.lift (P.map π.op) ⋯)
    -/
    have h := hP.bijective_mapToEqualizer_pullback _ X B π
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      hP : CategoryTheory.regularTopology.EqualizerCondition P
      X B : C
      π : Quiver.Hom X B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      h : Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Cat …
      ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.lift (P.map π.op) ⋯)
    -/
    rw [← isIso_iff_bijective, mapToEqualizer_eq_comp] at h
    exact IsIso.of_isIso_comp_right (equalizer.lift (P.map π.op)
      (equalizerCondition_w' P π))
      (Types.equalizerIso _ _).hom
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      ⊢ (∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [in …
    -/
  · intro hP
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
      ⊢ CategoryTheory.regularTopology.EqualizerCondition P
    -/
    apply EqualizerCondition.mk
    /-
      case mpr.hP
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
      ⊢ ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ins …
    -/
    intro X B π _ _
    /-
      case mpr.hP
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
      X B : C
      π : Quiver.Hom X B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      ⊢ Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer P π (Categ …
    -/
    rw [mapToEqualizer_eq_comp, ← isIso_iff_bijective]
    /-
      case mpr.hP
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      P : CategoryTheory.Functor (Opposite C) (Type u_4)
      hP : ∀ (X B : C) (π : Quiver.Hom X B) [inst : CategoryTheory.EffectiveEpi π] [ …
      X B : C
      π : Quiver.Hom X B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- `P` satisfies the equalizer condition iff its precomposition by an equivalence does. -/
theorem equalizerCondition_iff_of_equivalence (P : Cᵒᵖ ⥤ D)
    (e : C ≌ E) : EqualizerCondition P ↔ EqualizerCondition (e.op.inverse ⋙ P) :=
  ⟨fun h ↦ equalizerCondition_precomp_of_preservesPullback P e.inverse h, fun h ↦
    equalizerCondition_of_natIso (e.op.funInvIdAssoc P)
      (equalizerCondition_precomp_of_preservesPullback (e.op.inverse ⋙ P) e.functor h)⟩


open WalkingParallelPair WalkingParallelPairHom in
theorem parallelPair_pullback_initial {X B : C} (π : X ⟶ B)
    (c : PullbackCone π π) (hc : IsLimit c) :
    (parallelPair (C := (Sieve.ofArrows (fun (_ : Unit) => X) (fun _ => π)).arrows.categoryᵒᵖ)
    (Y := op ((Presieve.categoryMk _ (c.fst ≫ π) ⟨_, c.fst, π, ofArrows.mk (), rfl⟩)))
    (X := op ((Presieve.categoryMk _ π (Sieve.ofArrows_mk _ _ Unit.unit))))
                    /-
                      C : Type u_1
                      D : Type u_2
                      E : Type u_3
                      inst✝² : CategoryTheory.Category.{?u.55267, u_1} C
                      inst✝¹ : CategoryTheory.Category.{?u.55271, u_2} D
                      inst✝ : CategoryTheory.Category.{?u.55275, u_3} E
                      X B : C
                      π : Quiver.Hom X B
                      c : CategoryTheory.Limits.PullbackCone π π
                      hc : CategoryTheory.Limits.IsLimit c
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.fst ((CategoryTheory.Sieve.ofArrows …
                    -/
    (Quiver.Hom.op (Over.homMk c.fst))
                    /-
                      🎉 no goals
                    -/
    (Quiver.Hom.op (Over.homMk c.snd c.condition.symm))).Initial := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_4, u_1} C
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ (CategoryTheory.Limits.parallelPair (CategoryTheory.Over.homMk c.fst ⋯).op ( …
  -/
  apply Limits.parallelPair_initial_mk
    /-
      case h₁
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ (Z : Opposite (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arro …
    -/
  · intro ⟨Z⟩
    /-
      case h₁
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      ⊢ Nonempty (Quiver.Hom { unop := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    -/
    obtain ⟨_, f, g, ⟨⟩, hh⟩ := Z.property
    let X' : (Presieve.ofArrows (fun () ↦ X) (fun () ↦ π)).category :=
      Presieve.categoryMk _ π (ofArrows.mk ())
    /-
      case h₁.intro.intro.intro.intro.mk
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      Y : C
      i✝ : Unit
      f : Quiver.Hom ((CategoryTheory.Functor.id C).obj Z.obj.left) X
      hh : Eq (CategoryTheory.CategoryStruct.comp f π) Z.obj.hom
      X' : (CategoryTheory.Presieve.ofArrows (fun x => X) fun x => CategoryTheory.re …
      ⊢ Nonempty (Quiver.Hom { unop := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    -/
    let f' : Z.obj.left ⟶ X'.obj.left := f
    /-
      case h₁.intro.intro.intro.intro.mk
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      Y : C
      i✝ : Unit
      f : Quiver.Hom ((CategoryTheory.Functor.id C).obj Z.obj.left) X
      hh : Eq (CategoryTheory.CategoryStruct.comp f π) Z.obj.hom
      X' : (CategoryTheory.Presieve.ofArrows (fun x => X) fun x => CategoryTheory.re …
      f' : Quiver.Hom Z.obj.left X'.obj.left := f
      ⊢ Nonempty (Quiver.Hom { unop := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    -/
    exact ⟨(Over.homMk f').op⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ ⦃Z : Opposite (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arro …
    -/
  · intro ⟨Z⟩ ⟨i⟩ ⟨j⟩
    /-
      case h₂
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      i j : Quiver.Hom (Opposite.unop { unop := Z }) (Opposite.unop { unop := (Categ …
      ⊢ Exists fun a => And (Eq { unop := i } (CategoryTheory.CategoryStruct.comp (C …
    -/
    let ij := PullbackCone.IsLimit.lift hc i.left j.left (by erw [i.w, j.w]; rfl)
    /-
      case h₂
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      i j : Quiver.Hom (Opposite.unop { unop := Z }) (Opposite.unop { unop := (Categ …
      ij : Quiver.Hom (Opposite.unop { unop := Z }).obj.left c.pt := CategoryTheory. …
      ⊢ Exists fun a => And (Eq { unop := i } (CategoryTheory.CategoryStruct.comp (C …
    -/
    refine ⟨Quiver.Hom.op (Over.homMk ij (by simpa [ij] using i.w)), ?_, ?_⟩
    /-
      case h₂.refine_1
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      i j : Quiver.Hom (Opposite.unop { unop := Z }) (Opposite.unop { unop := (Categ …
      ij : Quiver.Hom (Opposite.unop { unop := Z }).obj.left c.pt := CategoryTheory. …
      ⊢ Eq { unop := i } (CategoryTheory.CategoryStruct.comp (CategoryTheory.Over.ho …
    -/
    all_goals congr
    /-
      case h₂.refine_1.e_unop
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_4, u_1} C
      X B : C
      π : Quiver.Hom X B
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      Z : (CategoryTheory.Sieve.ofArrows (fun x => X) fun x => π).arrows.category
      i j : Quiver.Hom (Opposite.unop { unop := Z }) (Opposite.unop { unop := (Categ …
      ij : Quiver.Hom (Opposite.unop { unop := Z }).obj.left c.pt := CategoryTheory. …
      ⊢ Eq i (CategoryTheory.CategoryStruct.comp (CategoryTheory.Over.homMk ij ⋯).op …
    -/
    all_goals exact Comma.hom_ext _ _ (by erw [Over.comp_left]; simp [ij]) rfl
    /-
      🎉 no goals
    -/


/--
Given a limiting pullback cone, the fork in `SingleEqualizerCondition` is limiting iff the diagram
in `Presheaf.isSheaf_iff_isLimit_coverage` is limiting.
-/
noncomputable def isLimit_forkOfι_equiv (P : Cᵒᵖ ⥤ D) {X B : C} (π : X ⟶ B)
    (c : PullbackCone π π) (hc : IsLimit c) :
    IsLimit (Fork.ofι (P.map π.op) (equalizerCondition_w P c)) ≃
    IsLimit (P.mapCone (Sieve.ofArrows (fun (_ : Unit) ↦ X) fun _ ↦ π).arrows.cocone.op) := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let S := (Sieve.ofArrows (fun (_ : Unit) => X) (fun _ => π)).arrows
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let X' := S.categoryMk π ⟨_, 𝟙 _, π, ofArrows.mk (), Category.id_comp _⟩
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let P' := S.categoryMk (c.fst ≫ π) ⟨_, c.fst, π, ofArrows.mk (), rfl⟩
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let fst : P' ⟶ X' := Over.homMk c.fst
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let snd : P' ⟶ X' := Over.homMk c.snd c.condition.symm
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let F : S.categoryᵒᵖ ⥤ D := S.diagram.op ⋙ P
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let G := parallelPair (P.map c.fst.op) (P.map c.snd.op)
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let H := parallelPair fst.op snd.op
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    H : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Opposite …
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  have : H.Initial := parallelPair_pullback_initial π c hc
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    H : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Opposite …
    this : H.Initial
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  let i : H ⋙ F ≅ G := parallelPair.ext (Iso.refl _) (Iso.refl _) (by aesop) (by aesop)
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    H : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Opposite …
    this : H.Initial
    i : CategoryTheory.Iso (H.comp F) G := CategoryTheory.Limits.parallelPair.ext  …
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (P.map  …
  -/
  refine (IsLimit.equivOfNatIsoOfIso i.symm _ _ ?_).trans (Functor.Initial.isLimitWhiskerEquiv H _)
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    H : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Opposite …
    this : H.Initial
    i : CategoryTheory.Iso (H.comp F) G := CategoryTheory.Limits.parallelPair.ext  …
    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose i.symm.hom).obj …
  -/
  refine Cones.ext (Iso.refl _) ?_
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    H : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Opposite …
    this : H.Initial
    i : CategoryTheory.Iso (H.comp F) G := CategoryTheory.Limits.parallelPair.ext  …
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (((CategoryTheory.Limi …
  -/
  rintro ⟨_ | _⟩
  /-
    case zero
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝² : CategoryTheory.Category.{?u.71842, u_1} C
    inst✝¹ : CategoryTheory.Category.{?u.71846, u_2} D
    inst✝ : CategoryTheory.Category.{?u.71850, u_3} E
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    S : CategoryTheory.Presieve B := (CategoryTheory.Sieve.ofArrows (fun x => X) f …
    X' : S.category := S.categoryMk π ⋯
    P' : S.category := S.categoryMk (CategoryTheory.CategoryStruct.comp c.fst π) ⋯
    fst : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.fst ⋯
    snd : Quiver.Hom P' X' := CategoryTheory.Over.homMk c.snd ⋯
    F : CategoryTheory.Functor (Opposite S.category) D := S.diagram.op.comp P
    G : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair D := Cate …
    H : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Opposite …
    this : H.Initial
    i : CategoryTheory.Iso (H.comp F) G := CategoryTheory.Limits.parallelPair.ext  …
    ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose i.symm.hom).obj (CategoryTheor …
  -/
  all_goals aesop
  /-
    🎉 no goals
  -/


lemma equalizerConditionMap_iff_nonempty_isLimit (P : Cᵒᵖ ⥤ D) ⦃X B : C⦄ (π : X ⟶ B)
    [HasPullback π π] : SingleEqualizerCondition P π ↔
      Nonempty (IsLimit (P.mapCone
        (Sieve.ofArrows (fun (_ : Unit) => X) (fun _ => π)).arrows.cocone.op)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    P : CategoryTheory.Functor (Opposite C) D
    X B : C
    π : Quiver.Hom X B
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Iff (CategoryTheory.regularTopology.SingleEqualizerCondition P π) (Nonempty  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      P : CategoryTheory.Functor (Opposite C) D
      X B : C
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.Limits.HasPullback π π
      ⊢ CategoryTheory.regularTopology.SingleEqualizerCondition P π → Nonempty (Cate …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      P : CategoryTheory.Functor (Opposite C) D
      X B : C
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.Limits.HasPullback π π
      h : CategoryTheory.regularTopology.SingleEqualizerCondition P π
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (P.mapCone (CategoryTheory.Sieve.ofA …
    -/
    exact ⟨isLimit_forkOfι_equiv _ _ _ (pullbackIsPullback π π) (h _ (pullbackIsPullback π π)).some⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      P : CategoryTheory.Functor (Opposite C) D
      X B : C
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.Limits.HasPullback π π
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (P.mapCone (CategoryTheory.Sieve.ofA …
    -/
  · intro ⟨h⟩
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      P : CategoryTheory.Functor (Opposite C) D
      X B : C
      π : Quiver.Hom X B
      inst✝ : CategoryTheory.Limits.HasPullback π π
      h : CategoryTheory.Limits.IsLimit (P.mapCone (CategoryTheory.Sieve.ofArrows (f …
      ⊢ CategoryTheory.regularTopology.SingleEqualizerCondition P π
    -/
    exact fun c hc ↦ ⟨(isLimit_forkOfι_equiv _ _ _ hc).symm h⟩
    /-
      🎉 no goals
    -/


lemma equalizerCondition_iff_isSheaf (F : Cᵒᵖ ⥤ D) [Preregular C]
    [∀ {Y X : C} (f : Y ⟶ X) [EffectiveEpi f], HasPullback f f] :
    EqualizerCondition F ↔ Presheaf.IsSheaf (regularTopology C) F := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor (Opposite C) D
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f …
    ⊢ Iff (CategoryTheory.regularTopology.EqualizerCondition F) (CategoryTheory.Pr …
  -/
  dsimp [regularTopology]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor (Opposite C) D
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f …
    ⊢ Iff (CategoryTheory.regularTopology.EqualizerCondition F) (CategoryTheory.Pr …
  -/
  rw [Presheaf.isSheaf_iff_isLimit_coverage]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor (Opposite C) D
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f …
    ⊢ Iff (CategoryTheory.regularTopology.EqualizerCondition F) (∀ ⦃X : C⦄ (R : Ca …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      F : CategoryTheory.Functor (Opposite C) D
      inst✝¹ : CategoryTheory.Preregular C
      inst✝ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f …
      ⊢ CategoryTheory.regularTopology.EqualizerCondition F → ∀ ⦃X : C⦄ (R : Categor …
    -/
  · rintro hF X _ ⟨Y, f, rfl, _⟩
    /-
      case mp.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      F : CategoryTheory.Functor (Opposite C) D
      inst✝¹ : CategoryTheory.Preregular C
      inst✝ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f …
      hF : CategoryTheory.regularTopology.EqualizerCondition F
      X Y : C
      f : Quiver.Hom Y X
      right✝ : CategoryTheory.EffectiveEpi f
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone (CategoryTheory.Sieve.gen …
    -/
    exact (equalizerConditionMap_iff_nonempty_isLimit F f).1 (hF f)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      F : CategoryTheory.Functor (Opposite C) D
      inst✝¹ : CategoryTheory.Preregular C
      inst✝ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f …
      ⊢ (∀ ⦃X : C⦄ (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheory. …
    -/
  · intro hF Y X f _
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Category.{u_5, u_2} D
      F : CategoryTheory.Functor (Opposite C) D
      inst✝² : CategoryTheory.Preregular C
      inst✝¹ : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi  …
      hF : ∀ ⦃X : C⦄ (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
      Y X : C
      f : Quiver.Hom Y X
      inst✝ : CategoryTheory.EffectiveEpi f
      ⊢ CategoryTheory.regularTopology.SingleEqualizerCondition F f
    -/
    exact (equalizerConditionMap_iff_nonempty_isLimit F f).2 (hF _ ⟨_, f, rfl, inferInstance⟩)
    /-
      🎉 no goals
    -/


lemma isSheafFor_regular_of_projective {X : C} (S : Presieve X) [S.regular] [Projective X]
    (F : Cᵒᵖ ⥤ Type*) : S.IsSheafFor F := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    X : C
    S : CategoryTheory.Presieve X
    inst✝¹ : S.regular
    inst✝ : CategoryTheory.Projective X
    F : CategoryTheory.Functor (Opposite C) (Type u_4)
    ⊢ CategoryTheory.Presieve.IsSheafFor F S
  -/
  obtain ⟨Y, f, rfl, hf⟩ := Presieve.regular.single_epi (R := S)
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    X : C
    inst✝¹ : CategoryTheory.Projective X
    F : CategoryTheory.Functor (Opposite C) (Type u_4)
    Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    inst✝ : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows (fun  …
  -/
  rw [isSheafFor_arrows_iff]
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    X : C
    inst✝¹ : CategoryTheory.Projective X
    F : CategoryTheory.Functor (Opposite C) (Type u_4)
    Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    inst✝ : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    ⊢ ∀ (x : Unit → F.obj { unop := Y }), CategoryTheory.Presieve.Arrows.Compatibl …
  -/
  refine fun x hx ↦ ⟨F.map (Projective.factorThru (𝟙 _) f).op <| x (), fun _ ↦ ?_, fun y h ↦ ?_⟩
    /-
      case intro.intro.intro.refine_1
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_5, u_1} C
      X : C
      inst✝¹ : CategoryTheory.Projective X
      F : CategoryTheory.Functor (Opposite C) (Type u_4)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.EffectiveEpi f
      inst✝ : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
      x : Unit → F.obj { unop := Y }
      hx : CategoryTheory.Presieve.Arrows.Compatible F (fun x => f) x
      x✝ : Unit
      ⊢ Eq (F.map f.op (F.map (CategoryTheory.Projective.factorThru (CategoryTheory. …
    -/
  · simpa using (hx () () Y (𝟙 Y) (f ≫ (Projective.factorThru (𝟙 _) f)) (by simp)).symm
    /-
      🎉 no goals
    -/
  · simp only [← h (), ← FunctorToTypes.map_comp_apply, ← op_comp, Projective.factorThru_comp,
      op_id, FunctorToTypes.map_id_apply]


/-- Every presheaf is a sheaf for the regular topology if every object of `C` is projective. -/
theorem isSheaf_of_projective (F : Cᵒᵖ ⥤ D) [Preregular C] [∀ (X : C), Projective X] :
    Presheaf.IsSheaf (regularTopology C) F :=
  fun _ ↦ (isSheaf_coverage _ _).mpr fun S ⟨_, h⟩ ↦ have : S.regular := ⟨_, h⟩
    isSheafFor_regular_of_projective _ _


/-- Every Yoneda-presheaf is a sheaf for the regular topology. -/
lemma isSheaf_yoneda_obj [Preregular C] (W : C)  :
    Presieve.IsSheaf (regularTopology C) (yoneda.obj W) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W : C
    ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.regularTopology C) (Category …
  -/
  rw [regularTopology, isSheaf_coverage]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W : C
    ⊢ ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheory.r …
  -/
  intro X S ⟨_, hS⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X : C
    S : CategoryTheory.Presieve X
    w✝ : C
    hS : Exists fun f => And (Eq S (CategoryTheory.Presieve.ofArrows (fun x => w✝) …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) S
  -/
  have : S.regular := ⟨_, hS⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X : C
    S : CategoryTheory.Presieve X
    w✝ : C
    hS : Exists fun f => And (Eq S (CategoryTheory.Presieve.ofArrows (fun x => w✝) …
    this : S.regular
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) S
  -/
  obtain ⟨Y, f, rfl, hf⟩ := Presieve.regular.single_epi (R := S)
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) (CategoryTh …
  -/
  have h_colim := isColimitOfEffectiveEpiStruct f hf.effectiveEpi.some
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generateSingle …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) (CategoryTh …
  -/
  rw [← Sieve.generateSingleton_eq, ← Presieve.ofArrows_pUnit] at h_colim
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) (CategoryTh …
  -/
  intro x hx
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let x_ext := Presieve.FamilyOfElements.sieveExtend x
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  have hx_ext := Presieve.FamilyOfElements.Compatible.sieveExtend hx
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
    hx_ext : x.sieveExtend.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let S := Sieve.generate (Presieve.ofArrows (fun () ↦ Y) (fun () ↦ f))
  obtain ⟨t, t_amalg, t_uniq⟩ :=
    (Sieve.forallYonedaIsSheaf_iff_colimit S).mpr ⟨h_colim⟩ W x_ext hx_ext
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Preregular C
    W X w✝ Y : C
    f : Quiver.Hom Y X
    hf : CategoryTheory.EffectiveEpi f
    hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
    this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
    hx_ext : x.sieveExtend.Compatible
    S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
    t : (CategoryTheory.yoneda.obj W).obj { unop := X }
    t_amalg : x_ext.IsAmalgamation t
    t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  refine ⟨t, ?_, ?_⟩
  · convert Presieve.isAmalgamation_restrict (Sieve.le_generate
      (Presieve.ofArrows (fun () ↦ Y) (fun () ↦ f))) _ _ t_amalg
    /-
      case h.e.h.e'_6.h.h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝ : CategoryTheory.Preregular C
      W X w✝ Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.EffectiveEpi f
      hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
      this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
      h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
      hx : x.Compatible
      x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
      hx_ext : x.sieveExtend.Compatible
      S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
      t : (CategoryTheory.yoneda.obj W).obj { unop := X }
      t_amalg : x_ext.IsAmalgamation t
      t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
      ⊢ Eq x (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x_ext)
    -/
    exact (Presieve.restrict_extend hx).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝ : CategoryTheory.Preregular C
      W X w✝ Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.EffectiveEpi f
      hS : Exists fun f_1 => And (Eq (CategoryTheory.Presieve.ofArrows (fun x => Y)  …
      this : (CategoryTheory.Presieve.ofArrows (fun x => Y) fun x => f).regular
      h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
      hx : x.Compatible
      x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
      hx_ext : x.sieveExtend.Compatible
      S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
      t : (CategoryTheory.yoneda.obj W).obj { unop := X }
      t_amalg : x_ext.IsAmalgamation t
      t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
      ⊢ ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x.IsAmalg …
    -/
  · exact fun y hy ↦ t_uniq y <| Presieve.isAmalgamation_sieveExtend x y hy
    /-
      🎉 no goals
    -/


/-- The regular topology on any preregular category is subcanonical. -/
instance subcanonical [Preregular C] : (regularTopology C).Subcanonical :=
  GrothendieckTopology.Subcanonical.of_isSheaf_yoneda_obj _ isSheaf_yoneda_obj


