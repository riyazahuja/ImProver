instance TopCat.of.chartedSpace : ChartedSpace H (TopCat.of M) :=
  (inferInstance : ChartedSpace H M)


instance TopCat.of.hasGroupoid [HasGroupoid M G] : HasGroupoid (TopCat.of M) G :=
  (inferInstance : HasGroupoid M G)


/-- Let `P` be a `LocalInvariantProp` for functions between spaces with the groupoids `G`, `G'`
and let `M`, `M'` be charted spaces modelled on the model spaces of those groupoids.  Then there is
an induced `LocalPredicate` on the functions from `M` to `M'`, given by `LiftProp P`. -/
def StructureGroupoid.LocalInvariantProp.localPredicate (hG : LocalInvariantProp G G' P) :
    TopCat.LocalPredicate fun _ : TopCat.of M => M' where
  pred {U : Opens (TopCat.of M)} := fun f : U → M' => ChartedSpace.LiftProp P f
  res := by
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      ⊢ ∀ {U V : TopologicalSpace.Opens ↑(TopCat.of M)} (i : Quiver.Hom U V) (f : (S …
    -/
    intro U V i f h x
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      U V : TopologicalSpace.Opens ↑(TopCat.of M)
      i : Quiver.Hom U V
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ChartedSpace.LiftProp P f
      x : Subtype fun x => Membership.mem U x
      ⊢ ChartedSpace.LiftPropAt P (fun x => f ((fun x => ⟨↑x, ⋯⟩) x)) x
    -/
    have hUV : U ≤ V := CategoryTheory.leOfHom i
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      U V : TopologicalSpace.Opens ↑(TopCat.of M)
      i : Quiver.Hom U V
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ChartedSpace.LiftProp P f
      x : Subtype fun x => Membership.mem U x
      hUV : LE.le U V
      ⊢ ChartedSpace.LiftPropAt P (fun x => f ((fun x => ⟨↑x, ⋯⟩) x)) x
    -/
    show ChartedSpace.LiftPropAt P (f ∘ Opens.inclusion hUV) x
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      U V : TopologicalSpace.Opens ↑(TopCat.of M)
      i : Quiver.Hom U V
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ChartedSpace.LiftProp P f
      x : Subtype fun x => Membership.mem U x
      hUV : LE.le U V
      ⊢ ChartedSpace.LiftPropAt P (Function.comp f (TopologicalSpace.Opens.inclusion …
    -/
    rw [← hG.liftPropAt_iff_comp_inclusion hUV]
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      U V : TopologicalSpace.Opens ↑(TopCat.of M)
      i : Quiver.Hom U V
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ChartedSpace.LiftProp P f
      x : Subtype fun x => Membership.mem U x
      hUV : LE.le U V
      ⊢ ChartedSpace.LiftPropAt P f (Set.inclusion hUV x)
    -/
    apply h
    /-
      🎉 no goals
    -/
  locality := by
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      ⊢ ∀ {U : TopologicalSpace.Opens ↑(TopCat.of M)} (f : (Subtype fun x => Members …
    -/
    intro V f h x
    /-
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      V : TopologicalSpace.Opens ↑(TopCat.of M)
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ∀ (x : Subtype fun x => Membership.mem V x), Exists fun V_1 => Exists fun  …
      x : Subtype fun x => Membership.mem V x
      ⊢ ChartedSpace.LiftPropAt P f x
    -/
    obtain ⟨U, hxU, i, hU : ChartedSpace.LiftProp P (f ∘ i)⟩ := h x
    /-
      case intro.intro.intro
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      V : TopologicalSpace.Opens ↑(TopCat.of M)
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ∀ (x : Subtype fun x => Membership.mem V x), Exists fun V_1 => Exists fun  …
      x : Subtype fun x => Membership.mem V x
      U : TopologicalSpace.Opens ↑(TopCat.of M)
      hxU : Membership.mem U ↑x
      i : Quiver.Hom U V
      hU : ChartedSpace.LiftProp P (Function.comp f fun x => ⟨↑x, ⋯⟩)
      ⊢ ChartedSpace.LiftPropAt P f x
    -/
    let x' : U := ⟨x, hxU⟩
    /-
      case intro.intro.intro
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      V : TopologicalSpace.Opens ↑(TopCat.of M)
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ∀ (x : Subtype fun x => Membership.mem V x), Exists fun V_1 => Exists fun  …
      x : Subtype fun x => Membership.mem V x
      U : TopologicalSpace.Opens ↑(TopCat.of M)
      hxU : Membership.mem U ↑x
      i : Quiver.Hom U V
      hU : ChartedSpace.LiftProp P (Function.comp f fun x => ⟨↑x, ⋯⟩)
      x' : Subtype fun x => Membership.mem U x := ⟨↑x, hxU⟩
      ⊢ ChartedSpace.LiftPropAt P f x
    -/
    have hUV : U ≤ V := CategoryTheory.leOfHom i
    have : ChartedSpace.LiftPropAt P f (Opens.inclusion hUV x') := by
      rw [hG.liftPropAt_iff_comp_inclusion hUV]
      exact hU x'
    /-
      case intro.intro.intro
      H : Type u_1
      inst✝⁵ : TopologicalSpace H
      H' : Type u_2
      inst✝⁴ : TopologicalSpace H'
      G : StructureGroupoid H
      G' : StructureGroupoid H'
      P : (H → H') → Set H → H → Prop
      M : Type u
      inst✝³ : TopologicalSpace M
      inst✝² : ChartedSpace H M
      M' : Type u
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      hG : G.LocalInvariantProp G' P
      V : TopologicalSpace.Opens ↑(TopCat.of M)
      f : (Subtype fun x => Membership.mem V x) → M'
      h : ∀ (x : Subtype fun x => Membership.mem V x), Exists fun V_1 => Exists fun  …
      x : Subtype fun x => Membership.mem V x
      U : TopologicalSpace.Opens ↑(TopCat.of M)
      hxU : Membership.mem U ↑x
      i : Quiver.Hom U V
      hU : ChartedSpace.LiftProp P (Function.comp f fun x => ⟨↑x, ⋯⟩)
      x' : Subtype fun x => Membership.mem U x := ⟨↑x, hxU⟩
      hUV : LE.le U V
      this : ChartedSpace.LiftPropAt P f (TopologicalSpace.Opens.inclusion hUV x')
      ⊢ ChartedSpace.LiftPropAt P f x
    -/
    convert this
    /-
      🎉 no goals
    -/


/-- Let `P` be a `LocalInvariantProp` for functions between spaces with the groupoids `G`, `G'`
and let `M`, `M'` be charted spaces modelled on the model spaces of those groupoids.  Then there is
a sheaf of types on `M` which, to each open set `U` in `M`, associates the type of bundled
functions from `U` to `M'` satisfying the lift of `P`. -/
def StructureGroupoid.LocalInvariantProp.sheaf (hG : LocalInvariantProp G G' P) :
    TopCat.Sheaf (Type u) (TopCat.of M) :=
  TopCat.subsheafToTypes (hG.localPredicate M M')


instance StructureGroupoid.LocalInvariantProp.sheafHasCoeToFun (hG : LocalInvariantProp G G' P)
    (U : (Opens (TopCat.of M))ᵒᵖ) : CoeFun ((hG.sheaf M M').val.obj U) fun _ => ↑(unop U) → M' where
  coe a := a.1


theorem StructureGroupoid.LocalInvariantProp.section_spec (hG : LocalInvariantProp G G' P)
    (U : (Opens (TopCat.of M))ᵒᵖ) (f : (hG.sheaf M M').val.obj U) : ChartedSpace.LiftProp P f :=
  f.2

