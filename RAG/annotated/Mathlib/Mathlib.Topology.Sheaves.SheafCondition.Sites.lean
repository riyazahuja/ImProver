/-- Given a presieve `R` on `U`, we obtain a covering family of open sets in `X`, by taking as index
type the type of dependent pairs `(V, f)`, where `f : V ⟶ U` is in `R`.
-/
def coveringOfPresieve (U : Opens X) (R : Presieve U) : (ΣV, { f : V ⟶ U // R f }) → Opens X :=
  fun f => f.1


@[simp]
theorem coveringOfPresieve_apply (U : Opens X) (R : Presieve U) (f : ΣV, { f : V ⟶ U // R f }) :
    coveringOfPresieve U R f = f.1 := rfl


/-- If `R` is a presieve in the grothendieck topology on `Opens X`, the covering family associated
to `R` really is _covering_, i.e. the union of all open sets equals `U`.
-/
theorem iSup_eq_of_mem_grothendieck (hR : Sieve.generate R ∈ Opens.grothendieckTopology X U) :
    iSup (coveringOfPresieve U R) = U := by
  /-
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve U
    hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
    ⊢ Eq (iSup (TopCat.Presheaf.coveringOfPresieve U R)) U
  -/
  apply le_antisymm
    /-
      case a
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      R : CategoryTheory.Presieve U
      hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
      ⊢ LE.le (iSup (TopCat.Presheaf.coveringOfPresieve U R)) U
    -/
  · refine iSup_le ?_
    /-
      case a
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      R : CategoryTheory.Presieve U
      hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
      ⊢ ∀ (i : Sigma fun V => Subtype fun f => R f), LE.le (TopCat.Presheaf.covering …
    -/
    intro f
    /-
      case a
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      R : CategoryTheory.Presieve U
      hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
      f : Sigma fun V => Subtype fun f => R f
      ⊢ LE.le (TopCat.Presheaf.coveringOfPresieve U R f) U
    -/
    exact f.2.1.le
    /-
      🎉 no goals
    -/
  /-
    case a
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve U
    hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
    ⊢ LE.le U (iSup (TopCat.Presheaf.coveringOfPresieve U R))
  -/
  intro x hxU
  /-
    case a
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve U
    hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
    x : ↑X
    hxU : Membership.mem (↑U) x
    ⊢ Membership.mem (↑(iSup (TopCat.Presheaf.coveringOfPresieve U R))) x
  -/
  rw [Opens.coe_iSup, Set.mem_iUnion]
  /-
    case a
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve U
    hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
    x : ↑X
    hxU : Membership.mem (↑U) x
    ⊢ Exists fun i => Membership.mem (↑(TopCat.Presheaf.coveringOfPresieve U R i)) x
  -/
  obtain ⟨V, iVU, ⟨W, iVW, iWU, hiWU, -⟩, hxV⟩ := hR x hxU
  /-
    case a.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve U
    hR : Membership.mem ((Opens.grothendieckTopology ↑X) U) (CategoryTheory.Sieve. …
    x : ↑X
    hxU : Membership.mem (↑U) x
    V : TopologicalSpace.Opens ↑X
    iVU : Quiver.Hom V U
    hxV : Membership.mem V x
    W : TopologicalSpace.Opens ↑X
    iVW : Quiver.Hom V W
    iWU : Quiver.Hom W U
    hiWU : R iWU
    ⊢ Exists fun i => Membership.mem (↑(TopCat.Presheaf.coveringOfPresieve U R i)) x
  -/
  exact ⟨⟨W, ⟨iWU, hiWU⟩⟩, iVW.le hxV⟩
  /-
    🎉 no goals
  -/


/-- Given a family of opens `U : ι → Opens X` and any open `Y : Opens X`, we obtain a presieve
on `Y` by declaring that a morphism `f : V ⟶ Y` is a member of the presieve if and only if
there exists an index `i : ι` such that `V = U i`.
-/
def presieveOfCoveringAux {ι : Type v} (U : ι → Opens X) (Y : Opens X) : Presieve Y :=
  fun V _ => ∃ i, V = U i


/-- Take `Y` to be `iSup U` and obtain a presieve over `iSup U`. -/
def presieveOfCovering {ι : Type v} (U : ι → Opens X) : Presieve (iSup U) :=
  presieveOfCoveringAux U (iSup U)


/-- Given a presieve `R` on `Y`, if we take its associated family of opens via
    `coveringOfPresieve` (which may not cover `Y` if `R` is not covering), and take
    the presieve on `Y` associated to the family of opens via `presieveOfCoveringAux`,
    then we get back the original presieve `R`. -/
@[simp]
theorem covering_presieve_eq_self {Y : Opens X} (R : Presieve Y) :
    presieveOfCoveringAux (coveringOfPresieve Y R) Y = R := by
  /-
    X : TopCat
    Y : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve Y
    ⊢ Eq (TopCat.Presheaf.presieveOfCoveringAux (TopCat.Presheaf.coveringOfPresiev …
  -/
  funext Z
  /-
    case h
    X : TopCat
    Y : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve Y
    Z : TopologicalSpace.Opens ↑X
    ⊢ Eq (TopCat.Presheaf.presieveOfCoveringAux (TopCat.Presheaf.coveringOfPresiev …
  -/
  ext f
  /-
    case h.h
    X : TopCat
    Y : TopologicalSpace.Opens ↑X
    R : CategoryTheory.Presieve Y
    Z : TopologicalSpace.Opens ↑X
    f : Quiver.Hom Z Y
    ⊢ Iff (Membership.mem (TopCat.Presheaf.presieveOfCoveringAux (TopCat.Presheaf. …
  -/
  exact ⟨fun ⟨⟨_, f', h⟩, rfl⟩ => by rwa [Subsingleton.elim f f'], fun h => ⟨⟨Z, f, h⟩, rfl⟩⟩
  /-
    🎉 no goals
  -/


/-- The sieve generated by `presieveOfCovering U` is a member of the grothendieck topology.
-/
theorem mem_grothendieckTopology :
    Sieve.generate (presieveOfCovering U) ∈ Opens.grothendieckTopology X (iSup U) := by
  /-
    X : TopCat
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    ⊢ Membership.mem ((Opens.grothendieckTopology ↑X) (iSup U)) (CategoryTheory.Si …
  -/
  intro x hx
  /-
    X : TopCat
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem (iSup U) x
    ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.generate (TopCa …
  -/
  obtain ⟨i, hxi⟩ := Opens.mem_iSup.mp hx
  /-
    case intro
    X : TopCat
    ι : Type v
    U : ι → TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem (iSup U) x
    i : ι
    hxi : Membership.mem (U i) x
    ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.generate (TopCa …
  -/
  exact ⟨U i, Opens.leSupr U i, ⟨U i, 𝟙 _, Opens.leSupr U i, ⟨i, rfl⟩, Category.id_comp _⟩, hxi⟩
  /-
    🎉 no goals
  -/


/-- An index `i : ι` can be turned into a dependent pair `(V, f)`, where `V` is an open set and
`f : V ⟶ iSup U` is a member of `presieveOfCovering U f`.
-/
def homOfIndex (i : ι) : ΣV, { f : V ⟶ iSup U // presieveOfCovering U f } :=
  ⟨U i, Opens.leSupr U i, i, rfl⟩


/-- By using the axiom of choice, a dependent pair `(V, f)` where `f : V ⟶ iSup U` is a member of
`presieveOfCovering U f` can be turned into an index `i : ι`, such that `V = U i`.
-/
def indexOfHom (f : ΣV, { f : V ⟶ iSup U // presieveOfCovering U f }) : ι :=
  f.2.2.choose


theorem indexOfHom_spec (f : ΣV, { f : V ⟶ iSup U // presieveOfCovering U f }) :
    f.1 = U (indexOfHom U f) :=
  f.2.2.choose_spec


theorem coverDense_iff_isBasis [Category ι] (B : ι ⥤ Opens X) :
    B.IsCoverDense (Opens.grothendieckTopology X) ↔ Opens.IsBasis (Set.range B.obj) := by
  /-
    X : TopCat
    ι : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} ι
    B : CategoryTheory.Functor ι (TopologicalSpace.Opens ↑X)
    ⊢ Iff (B.IsCoverDense (Opens.grothendieckTopology ↑X)) (TopologicalSpace.Opens …
  -/
  rw [Opens.isBasis_iff_nbhd]
  /-
    X : TopCat
    ι : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} ι
    B : CategoryTheory.Functor ι (TopologicalSpace.Opens ↑X)
    ⊢ Iff (B.IsCoverDense (Opens.grothendieckTopology ↑X)) (∀ {U : TopologicalSpac …
  -/
  constructor
    /-
      case mp
      X : TopCat
      ι : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} ι
      B : CategoryTheory.Functor ι (TopologicalSpace.Opens ↑X)
      ⊢ B.IsCoverDense (Opens.grothendieckTopology ↑X) → ∀ {U : TopologicalSpace.Ope …
    -/
  · intro hd U x hx; rcases hd.1 U x hx with ⟨V, f, ⟨i, f₁, f₂, _⟩, hV⟩
    /-
      case mp.intro.intro.intro.intro.mk
      X : TopCat
      ι : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} ι
      B : CategoryTheory.Functor ι (TopologicalSpace.Opens ↑X)
      hd : B.IsCoverDense (Opens.grothendieckTopology ↑X)
      U : TopologicalSpace.Opens ↑X
      x : ↑X
      hx : Membership.mem U x
      V : TopologicalSpace.Opens ↑X
      f : Quiver.Hom V U
      hV : Membership.mem V x
      i : ι
      f₁ : Quiver.Hom V (B.obj i)
      f₂ : Quiver.Hom (B.obj i) U
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f
      ⊢ Exists fun U' => And (Membership.mem (Set.range B.obj) U') (And (Membership. …
    -/
    exact ⟨B.obj i, ⟨i, rfl⟩, f₁.le hV, f₂.le⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    X : TopCat
    ι : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} ι
    B : CategoryTheory.Functor ι (TopologicalSpace.Opens ↑X)
    ⊢ (∀ {U : TopologicalSpace.Opens ↑X} {x : ↑X}, Membership.mem U x → Exists fun …
  -/
  intro hb; constructor; intro U x hx; rcases hb hx with ⟨_, ⟨i, rfl⟩, hx, hi⟩
  /-
    case mpr.is_cover.intro.intro.intro.intro
    X : TopCat
    ι : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} ι
    B : CategoryTheory.Functor ι (TopologicalSpace.Opens ↑X)
    hb : ∀ {U : TopologicalSpace.Opens ↑X} {x : ↑X}, Membership.mem U x → Exists f …
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    hx✝ : Membership.mem U x
    i : ι
    hx : Membership.mem (B.obj i) x
    hi : LE.le (B.obj i) U
    ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.coverByImage B  …
  -/
  exact ⟨B.obj i, ⟨⟨hi⟩⟩, ⟨⟨i, 𝟙 _, ⟨⟨hi⟩⟩, rfl⟩⟩, hx⟩
  /-
    🎉 no goals
  -/


theorem coverDense_inducedFunctor {B : ι → Opens X} (h : Opens.IsBasis (Set.range B)) :
    (inducedFunctor B).IsCoverDense (Opens.grothendieckTopology X)  :=
  (coverDense_iff_isBasis _).2 h


theorem Topology.IsOpenEmbedding.compatiblePreserving (hf : IsOpenEmbedding f) :
    CompatiblePreserving (Opens.grothendieckTopology Y) hf.isOpenMap.functor := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsOpenEmbedding ⇑f
    ⊢ CategoryTheory.CompatiblePreserving (Opens.grothendieckTopology ↑Y) ⋯.functor
  -/
  haveI : Mono f := (TopCat.mono_iff_injective f).mpr hf.injective
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    ⊢ CategoryTheory.CompatiblePreserving (Opens.grothendieckTopology ↑Y) ⋯.functor
  -/
  apply compatiblePreservingOfDownwardsClosed
  /-
    case hF
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    ⊢ {c : TopologicalSpace.Opens ↑X} → {d : TopologicalSpace.Opens ↑Y} → Quiver.H …
  -/
  intro U V i
  /-
    case hF
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    U : TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑Y
    i : Quiver.Hom V (⋯.functor.obj U)
    ⊢ Sigma fun c' => CategoryTheory.Iso (⋯.functor.obj c') V
  -/
  refine ⟨(Opens.map f).obj V, eqToIso <| Opens.ext <| Set.image_preimage_eq_of_subset fun x h ↦ ?_⟩
  /-
    case hF
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    U : TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑Y
    i : Quiver.Hom V (⋯.functor.obj U)
    x : ↑Y
    h : Membership.mem V.1 x
    ⊢ Membership.mem (Set.range ⇑f) x
  -/
  obtain ⟨_, _, rfl⟩ := i.le h
  /-
    case hF.intro.intro
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsOpenEmbedding ⇑f
    this : CategoryTheory.Mono f
    U : TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑Y
    i : Quiver.Hom V (⋯.functor.obj U)
    w✝ : ↑X
    left✝ : Membership.mem (↑U) w✝
    h : Membership.mem V.1 (f w✝)
    ⊢ Membership.mem (Set.range ⇑f) (f w✝)
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.compatiblePreserving := IsOpenEmbedding.compatiblePreserving


theorem IsOpenMap.coverPreserving (hf : IsOpenMap f) :
    CoverPreserving (Opens.grothendieckTopology X) (Opens.grothendieckTopology Y) hf.functor := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : IsOpenMap ⇑f
    ⊢ CategoryTheory.CoverPreserving (Opens.grothendieckTopology ↑X) (Opens.grothe …
  -/
  constructor
  /-
    case cover_preserve
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : IsOpenMap ⇑f
    ⊢ ∀ {U : TopologicalSpace.Opens ↑X} {S : CategoryTheory.Sieve U}, Membership.m …
  -/
  rintro U S hU _ ⟨x, hx, rfl⟩
  /-
    case cover_preserve.intro.intro
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : IsOpenMap ⇑f
    U : TopologicalSpace.Opens ↑X
    S : CategoryTheory.Sieve U
    hU : Membership.mem ((Opens.grothendieckTopology ↑X) U) S
    x : ↑X
    hx : Membership.mem (↑U) x
    ⊢ Exists fun U_1 => Exists fun f_1 => And ((CategoryTheory.Sieve.functorPushfo …
  -/
  obtain ⟨V, i, hV, hxV⟩ := hU x hx
  /-
    case cover_preserve.intro.intro.intro.intro.intro
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : IsOpenMap ⇑f
    U : TopologicalSpace.Opens ↑X
    S : CategoryTheory.Sieve U
    hU : Membership.mem ((Opens.grothendieckTopology ↑X) U) S
    x : ↑X
    hx : Membership.mem (↑U) x
    V : TopologicalSpace.Opens ↑X
    i : Quiver.Hom V U
    hV : S.arrows i
    hxV : Membership.mem V x
    ⊢ Exists fun U_1 => Exists fun f_1 => And ((CategoryTheory.Sieve.functorPushfo …
  -/
  exact ⟨_, hf.functor.map i, ⟨_, i, 𝟙 _, hV, rfl⟩, Set.mem_image_of_mem f hxV⟩
  /-
    🎉 no goals
  -/



lemma Topology.IsOpenEmbedding.functor_isContinuous (h : IsOpenEmbedding f) :
    h.isOpenMap.functor.IsContinuous (Opens.grothendieckTopology X)
      (Opens.grothendieckTopology Y) := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    h : Topology.IsOpenEmbedding ⇑f
    ⊢ ⋯.functor.IsContinuous (Opens.grothendieckTopology ↑X) (Opens.grothendieckTo …
  -/
  apply Functor.isContinuous_of_coverPreserving
    /-
      case hF₁
      X Y : TopCat
      f : Quiver.Hom X Y
      h : Topology.IsOpenEmbedding ⇑f
      ⊢ CategoryTheory.CompatiblePreserving (Opens.grothendieckTopology ↑Y) ⋯.functor
    -/
  · exact h.compatiblePreserving
    /-
      🎉 no goals
    -/
    /-
      case hF₂
      X Y : TopCat
      f : Quiver.Hom X Y
      h : Topology.IsOpenEmbedding ⇑f
      ⊢ CategoryTheory.CoverPreserving (Opens.grothendieckTopology ↑X) (Opens.grothe …
    -/
  · exact h.isOpenMap.coverPreserving
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.functor_isContinuous := IsOpenEmbedding.functor_isContinuous


theorem TopCat.Presheaf.isSheaf_of_isOpenEmbedding (h : IsOpenEmbedding f) (hF : F.IsSheaf) :
    IsSheaf (h.isOpenMap.functor.op ⋙ F) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    h : Topology.IsOpenEmbedding ⇑f
    hF : F.IsSheaf
    ⊢ TopCat.Presheaf.IsSheaf (⋯.functor.op.comp F)
  -/
  have := h.functor_isContinuous
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    h : Topology.IsOpenEmbedding ⇑f
    hF : F.IsSheaf
    this : ⋯.functor.IsContinuous (Opens.grothendieckTopology ↑X) (Opens.grothendi …
    ⊢ TopCat.Presheaf.IsSheaf (⋯.functor.op.comp F)
  -/
  exact Functor.op_comp_isSheaf _ _ _ ⟨_, hF⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias TopCat.Presheaf.isSheaf_of_openEmbedding := TopCat.Presheaf.isSheaf_of_isOpenEmbedding


instance : RepresentablyFlat (Opens.map f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    ⊢ CategoryTheory.RepresentablyFlat (TopologicalSpace.Opens.map f)
  -/
  constructor
  /-
    case cofiltered
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    ⊢ ∀ (X_1 : TopologicalSpace.Opens ↑X), CategoryTheory.IsCofiltered (CategoryTh …
  -/
  intro U
  /-
    case cofiltered
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    U : TopologicalSpace.Opens ↑X
    ⊢ CategoryTheory.IsCofiltered (CategoryTheory.StructuredArrow U (TopologicalSp …
  -/
  refine @IsCofiltered.mk _ _ ?_ ?_
    /-
      case cofiltered.refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      U : TopologicalSpace.Opens ↑X
      ⊢ CategoryTheory.IsCofilteredOrEmpty (CategoryTheory.StructuredArrow U (Topolo …
    -/
  · constructor
      /-
        case cofiltered.refine_1.cone_objs
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : TopCat
        f : Quiver.Hom X Y
        F : TopCat.Presheaf C Y
        U : TopologicalSpace.Opens ↑X
        ⊢ ∀ (X_1 Y_1 : CategoryTheory.StructuredArrow U (TopologicalSpace.Opens.map f) …
      -/
    · intro V W
      exact ⟨⟨⟨PUnit.unit⟩, V.right ⊓ W.right, homOfLE <| le_inf V.hom.le W.hom.le⟩,
        StructuredArrow.homMk (homOfLE inf_le_left),
        StructuredArrow.homMk (homOfLE inf_le_right), trivial⟩
      /-
        case cofiltered.refine_1.cone_maps
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : TopCat
        f : Quiver.Hom X Y
        F : TopCat.Presheaf C Y
        U : TopologicalSpace.Opens ↑X
        ⊢ ∀ ⦃X_1 Y_1 : CategoryTheory.StructuredArrow U (TopologicalSpace.Opens.map f) …
      -/
    · exact fun _ _ _ _ ↦ ⟨_, 𝟙 _, by simp [eq_iff_true_of_subsingleton]⟩
      /-
        🎉 no goals
      -/
    /-
      case cofiltered.refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      U : TopologicalSpace.Opens ↑X
      ⊢ Nonempty (CategoryTheory.StructuredArrow U (TopologicalSpace.Opens.map f))
    -/
  · exact ⟨StructuredArrow.mk <| show U ⟶ (Opens.map f).obj ⊤ from homOfLE le_top⟩
    /-
      🎉 no goals
    -/


theorem compatiblePreserving_opens_map :
    CompatiblePreserving (Opens.grothendieckTopology X) (Opens.map f) :=
  compatiblePreservingOfFlat _ _


theorem coverPreserving_opens_map : CoverPreserving (Opens.grothendieckTopology Y)
    (Opens.grothendieckTopology X) (Opens.map f) := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.CoverPreserving (Opens.grothendieckTopology ↑Y) (Opens.grothe …
  -/
  constructor
  /-
    case cover_preserve
    X Y : TopCat
    f : Quiver.Hom X Y
    ⊢ ∀ {U : TopologicalSpace.Opens ↑Y} {S : CategoryTheory.Sieve U}, Membership.m …
  -/
  intro U S hS x hx
  /-
    case cover_preserve
    X Y : TopCat
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑Y
    S : CategoryTheory.Sieve U
    hS : Membership.mem ((Opens.grothendieckTopology ↑Y) U) S
    x : ↑X
    hx : Membership.mem ((TopologicalSpace.Opens.map f).obj U) x
    ⊢ Exists fun U_1 => Exists fun f_1 => And ((CategoryTheory.Sieve.functorPushfo …
  -/
  obtain ⟨V, i, hi, hxV⟩ := hS (f x) hx
  /-
    case cover_preserve.intro.intro.intro
    X Y : TopCat
    f : Quiver.Hom X Y
    U : TopologicalSpace.Opens ↑Y
    S : CategoryTheory.Sieve U
    hS : Membership.mem ((Opens.grothendieckTopology ↑Y) U) S
    x : ↑X
    hx : Membership.mem ((TopologicalSpace.Opens.map f).obj U) x
    V : TopologicalSpace.Opens ↑Y
    i : Quiver.Hom V U
    hi : S.arrows i
    hxV : Membership.mem V (f x)
    ⊢ Exists fun U_1 => Exists fun f_1 => And ((CategoryTheory.Sieve.functorPushfo …
  -/
  exact ⟨_, (Opens.map f).map i, ⟨_, _, 𝟙 _, hi, Subsingleton.elim _ _⟩, hxV⟩
  /-
    🎉 no goals
  -/


instance : (Opens.map f).IsContinuous (Opens.grothendieckTopology Y)
    (Opens.grothendieckTopology X) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : TopCat
    f : Quiver.Hom X Y
    F : TopCat.Presheaf C Y
    ⊢ (TopologicalSpace.Opens.map f).IsContinuous (Opens.grothendieckTopology ↑Y)  …
  -/
  apply Functor.isContinuous_of_coverPreserving
    /-
      case hF₁
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      ⊢ CategoryTheory.CompatiblePreserving (Opens.grothendieckTopology ↑X) (Topolog …
    -/
  · exact compatiblePreserving_opens_map f
    /-
      🎉 no goals
    -/
    /-
      case hF₂
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : TopCat
      f : Quiver.Hom X Y
      F : TopCat.Presheaf C Y
      ⊢ CategoryTheory.CoverPreserving (Opens.grothendieckTopology ↑Y) (Opens.grothe …
    -/
  · exact coverPreserving_opens_map f
    /-
      🎉 no goals
    -/


/-- The empty component of a sheaf is terminal. -/
def isTerminalOfEmpty (F : Sheaf C X) : Limits.IsTerminal (F.val.obj (op ⊥)) :=
  F.isTerminalOfBotCover ⊥ (fun _ h => h.elim)


/-- A variant of `isTerminalOfEmpty` that is easier to `apply`. -/
def isTerminalOfEqEmpty (F : X.Sheaf C) {U : Opens X} (h : U = ⊥) :
    Limits.IsTerminal (F.val.obj (op U)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    ι : Type u_1
    B : ι → TopologicalSpace.Opens ↑X
    F✝ : TopCat.Presheaf C X
    F' F : TopCat.Sheaf C X
    U : TopologicalSpace.Opens ↑X
    h : Eq U Bot.bot
    ⊢ CategoryTheory.Limits.IsTerminal (F.val.obj { unop := U })
  -/
  convert F.isTerminalOfEmpty
  /-
    🎉 no goals
  -/


/-- If a family `B` of open sets forms a basis of the topology on `X`, and if `F'`
    is a sheaf on `X`, then a homomorphism between a presheaf `F` on `X` and `F'`
    is equivalent to a homomorphism between their restrictions to the indexing type
    `ι` of `B`, with the induced category structure on `ι`. -/
def restrictHomEquivHom (h : Opens.IsBasis (Set.range B)) :
    ((inducedFunctor B).op ⋙ F ⟶ (inducedFunctor B).op ⋙ F'.1) ≃ (F ⟶ F'.1) :=
  @Functor.IsCoverDense.restrictHomEquivHom _ _ _ _ _ _ _ _
    (Opens.coverDense_inducedFunctor h) _ F F'


@[simp]
theorem extend_hom_app (h : Opens.IsBasis (Set.range B))
    (α : (inducedFunctor B).op ⋙ F ⟶ (inducedFunctor B).op ⋙ F'.1) (i : ι) :
    (restrictHomEquivHom F F' h α).app (op (B i)) = α.app (op i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    ι : Type u_1
    B : ι → TopologicalSpace.Opens ↑X
    F : TopCat.Presheaf C X
    F' : TopCat.Sheaf C X
    h : TopologicalSpace.Opens.IsBasis (Set.range B)
    α : Quiver.Hom ((CategoryTheory.inducedFunctor B).op.comp F) ((CategoryTheory. …
    i : ι
    ⊢ Eq (((TopCat.Sheaf.restrictHomEquivHom F F' h) α).app { unop := B i }) (α.ap …
  -/
  nth_rw 2 [← (restrictHomEquivHom F F' h).left_inv α]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    ι : Type u_1
    B : ι → TopologicalSpace.Opens ↑X
    F : TopCat.Presheaf C X
    F' : TopCat.Sheaf C X
    h : TopologicalSpace.Opens.IsBasis (Set.range B)
    α : Quiver.Hom ((CategoryTheory.inducedFunctor B).op.comp F) ((CategoryTheory. …
    i : ι
    ⊢ Eq (((TopCat.Sheaf.restrictHomEquivHom F F' h) α).app { unop := B i }) (((To …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem hom_ext (h : Opens.IsBasis (Set.range B))
    {α β : F ⟶ F'.1} (he : ∀ i, α.app (op (B i)) = β.app (op (B i))) : α = β := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    ι : Type u_1
    B : ι → TopologicalSpace.Opens ↑X
    F : TopCat.Presheaf C X
    F' : TopCat.Sheaf C X
    h : TopologicalSpace.Opens.IsBasis (Set.range B)
    α β : Quiver.Hom F F'.val
    he : ∀ (i : ι), Eq (α.app { unop := B i }) (β.app { unop := B i })
    ⊢ Eq α β
  -/
  apply (restrictHomEquivHom F F' h).symm.injective
  /-
    case a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    ι : Type u_1
    B : ι → TopologicalSpace.Opens ↑X
    F : TopCat.Presheaf C X
    F' : TopCat.Sheaf C X
    h : TopologicalSpace.Opens.IsBasis (Set.range B)
    α β : Quiver.Hom F F'.val
    he : ∀ (i : ι), Eq (α.app { unop := B i }) (β.app { unop := B i })
    ⊢ Eq ((TopCat.Sheaf.restrictHomEquivHom F F' h).symm α) ((TopCat.Sheaf.restric …
  -/
  ext i
  /-
    case a.w.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : TopCat
    ι : Type u_1
    B : ι → TopologicalSpace.Opens ↑X
    F : TopCat.Presheaf C X
    F' : TopCat.Sheaf C X
    h : TopologicalSpace.Opens.IsBasis (Set.range B)
    α β : Quiver.Hom F F'.val
    he : ∀ (i : ι), Eq (α.app { unop := B i }) (β.app { unop := B i })
    i : Opposite (CategoryTheory.InducedCategory (TopologicalSpace.Opens ↑X) B)
    ⊢ Eq (((TopCat.Sheaf.restrictHomEquivHom F F' h).symm α).app i) (((TopCat.Shea …
  -/
  exact he i.unop
  /-
    🎉 no goals
  -/


