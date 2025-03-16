/-- An `IntegrationParams` is a structure holding 3 boolean values used to define a filter to be
used in the definition of a box-integrable function.

* `bRiemann`: the value `true` means that the filter corresponds to a Riemann-style integral, i.e.
  in the definition of integrability we require a constant upper estimate `r` on the size of boxes
  of a tagged partition; the value `false` means that the estimate may depend on the position of the
  tag.

* `bHenstock`: the value `true` means that we require that each tag belongs to its own closed box;
  the value `false` means that we only require that tags belong to the ambient box.

* `bDistortion`: the value `true` means that `r` can depend on the maximal ratio of sides of the
  same box of a partition. Presence of this case makes quite a few proofs harder but we can prove
  the divergence theorem only for the filter `BoxIntegral.IntegrationParams.GP = ⊥ =
  {bRiemann := false, bHenstock := true, bDistortion := true}`.
-/
@[ext]
structure IntegrationParams : Type where
  (bRiemann bHenstock bDistortion : Bool)


/-- Auxiliary equivalence with a product type used to lift an order. -/
def equivProd : IntegrationParams ≃ Bool × Boolᵒᵈ × Boolᵒᵈ where
  toFun l := ⟨l.1, OrderDual.toDual l.2, OrderDual.toDual l.3⟩
  invFun l := ⟨l.1, OrderDual.ofDual l.2.1, OrderDual.ofDual l.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl


instance : PartialOrder IntegrationParams :=
  PartialOrder.lift equivProd equivProd.injective


/-- Auxiliary `OrderIso` with a product type used to lift a `BoundedOrder` structure. -/
def isoProd : IntegrationParams ≃o Bool × Boolᵒᵈ × Boolᵒᵈ :=
  ⟨equivProd, Iff.rfl⟩


instance : BoundedOrder IntegrationParams :=
  isoProd.symm.toGaloisInsertion.liftBoundedOrder


/-- The value `BoxIntegral.IntegrationParams.GP = ⊥`
(`bRiemann = false`, `bHenstock = true`, `bDistortion = true`)
corresponds to a generalization of the Henstock integral such that the Divergence theorem holds true
without additional integrability assumptions, see the module docstring for details. -/
instance : Inhabited IntegrationParams :=
  ⟨⊥⟩


instance : DecidableRel ((· ≤ ·) : IntegrationParams → IntegrationParams → Prop) :=
  fun _ _ => inferInstanceAs (Decidable (_ ∧ _))


instance : DecidableEq IntegrationParams :=
  fun _ _ => decidable_of_iff _ IntegrationParams.ext_iff.symm


/-- The `BoxIntegral.IntegrationParams` corresponding to the Riemann integral. In the
corresponding filter, we require that the diameters of all boxes `J` of a tagged partition are
bounded from above by a constant upper estimate that may not depend on the geometry of `J`, and each
tag belongs to the corresponding closed box. -/
def Riemann : IntegrationParams where
  bRiemann := true
  bHenstock := true
  bDistortion := false


/-- The `BoxIntegral.IntegrationParams` corresponding to the Henstock-Kurzweil integral. In the
corresponding filter, we require that the tagged partition is subordinate to a (possibly,
discontinuous) positive function `r` and each tag belongs to the corresponding closed box. -/
def Henstock : IntegrationParams :=
  ⟨false, true, false⟩


/-- The `BoxIntegral.IntegrationParams` corresponding to the McShane integral. In the
corresponding filter, we require that the tagged partition is subordinate to a (possibly,
discontinuous) positive function `r`; the tags may be outside of the corresponding closed box
(but still inside the ambient closed box `I.Icc`). -/
def McShane : IntegrationParams :=
  ⟨false, false, false⟩


/-- The `BoxIntegral.IntegrationParams` corresponding to the generalized Perron integral. In the
corresponding filter, we require that the tagged partition is subordinate to a (possibly,
discontinuous) positive function `r` and each tag belongs to the corresponding closed box. We also
require an upper estimate on the distortion of all boxes of the partition. -/
def GP : IntegrationParams := ⊥


                                                       /-
                                                         ⊢ LE.le BoxIntegral.IntegrationParams.Henstock BoxIntegral.IntegrationParams.R …
                                                       -/
theorem henstock_le_riemann : Henstock ≤ Riemann := by trivial
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                       /-
                                                         ⊢ LE.le BoxIntegral.IntegrationParams.Henstock BoxIntegral.IntegrationParams.M …
                                                       -/
theorem henstock_le_mcShane : Henstock ≤ McShane := by trivial
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem gp_le : GP ≤ l :=
  bot_le


/-- The predicate corresponding to a base set of the filter defined by an
`IntegrationParams`. It says that

* if `l.bHenstock`, then `π` is a Henstock prepartition, i.e. each tag belongs to the corresponding
  closed box;
* `π` is subordinate to `r`;
* if `l.bDistortion`, then the distortion of each box in `π` is less than or equal to `c`;
* if `l.bDistortion`, then there exists a prepartition `π'` with distortion `≤ c` that covers
  exactly `I \ π.iUnion`.

The last condition is automatically verified for partitions, and is used in the proof of the
Sacks-Henstock inequality to compare two prepartitions covering the same part of the box.

It is also automatically satisfied for any `c > 1`, see TODO section of the module docstring for
details. -/
structure MemBaseSet (l : IntegrationParams) (I : Box ι) (c : ℝ≥0) (r : (ι → ℝ) → Ioi (0 : ℝ))
    (π : TaggedPrepartition I) : Prop where
  protected isSubordinate : π.IsSubordinate r
  protected isHenstock : l.bHenstock → π.IsHenstock
  protected distortion_le : l.bDistortion → π.distortion ≤ c
  protected exists_compl : l.bDistortion → ∃ π' : Prepartition I,
    π'.iUnion = ↑I \ π.iUnion ∧ π'.distortion ≤ c


/-- A predicate saying that in case `l.bRiemann = true`, the function `r` is a constant. -/
def RCond {ι : Type*} (l : IntegrationParams) (r : (ι → ℝ) → Ioi (0 : ℝ)) : Prop :=
  l.bRiemann → ∀ x, r x = r 0


/-- A set `s : Set (TaggedPrepartition I)` belongs to `l.toFilterDistortion I c` if there exists
a function `r : ℝⁿ → (0, ∞)` (or a constant `r` if `l.bRiemann = true`) such that `s` contains each
prepartition `π` such that `l.MemBaseSet I c r π`. -/
def toFilterDistortion (l : IntegrationParams) (I : Box ι) (c : ℝ≥0) :
    Filter (TaggedPrepartition I) :=
  ⨅ (r : (ι → ℝ) → Ioi (0 : ℝ)) (_ : l.RCond r), 𝓟 { π | l.MemBaseSet I c r π }


/-- A set `s : Set (TaggedPrepartition I)` belongs to `l.toFilter I` if for any `c : ℝ≥0` there
exists a function `r : ℝⁿ → (0, ∞)` (or a constant `r` if `l.bRiemann = true`) such that
`s` contains each prepartition `π` such that `l.MemBaseSet I c r π`. -/
def toFilter (l : IntegrationParams) (I : Box ι) : Filter (TaggedPrepartition I) :=
  ⨆ c : ℝ≥0, l.toFilterDistortion I c


/-- A set `s : Set (TaggedPrepartition I)` belongs to `l.toFilterDistortioniUnion I c π₀` if
there exists a function `r : ℝⁿ → (0, ∞)` (or a constant `r` if `l.bRiemann = true`) such that `s`
contains each prepartition `π` such that `l.MemBaseSet I c r π` and `π.iUnion = π₀.iUnion`. -/
def toFilterDistortioniUnion (l : IntegrationParams) (I : Box ι) (c : ℝ≥0) (π₀ : Prepartition I) :=
  l.toFilterDistortion I c ⊓ 𝓟 { π | π.iUnion = π₀.iUnion }


/-- A set `s : Set (TaggedPrepartition I)` belongs to `l.toFilteriUnion I π₀` if for any `c : ℝ≥0`
there exists a function `r : ℝⁿ → (0, ∞)` (or a constant `r` if `l.bRiemann = true`) such that `s`
contains each prepartition `π` such that `l.MemBaseSet I c r π` and `π.iUnion = π₀.iUnion`. -/
def toFilteriUnion (I : Box ι) (π₀ : Prepartition I) :=
  ⨆ c : ℝ≥0, l.toFilterDistortioniUnion I c π₀


theorem rCond_of_bRiemann_eq_false {ι} (l : IntegrationParams) (hl : l.bRiemann = false)
    {r : (ι → ℝ) → Ioi (0 : ℝ)} : l.RCond r := by
  /-
    ι : Type u_2
    l : BoxIntegral.IntegrationParams
    hl : Eq l.bRiemann Bool.false
    r : (ι → Real) → ↑(Set.Ioi 0)
    ⊢ l.RCond r
  -/
  simp [RCond, hl]
  /-
    🎉 no goals
  -/


theorem toFilter_inf_iUnion_eq (l : IntegrationParams) (I : Box ι) (π₀ : Prepartition I) :
    l.toFilter I ⊓ 𝓟 { π | π.iUnion = π₀.iUnion } = l.toFilteriUnion I π₀ :=
  (iSup_inf_principal _ _).symm


variable (I) in
theorem MemBaseSet.mono' (h : l₁ ≤ l₂) (hc : c₁ ≤ c₂)
    (hr : ∀ J ∈ π, r₁ (π.tag J) ≤ r₂ (π.tag J)) (hπ : l₁.MemBaseSet I c₁ r₁ π) :
    l₂.MemBaseSet I c₂ r₂ π :=
  ⟨hπ.1.mono' hr, fun h₂ => hπ.2 (le_iff_imp.1 h.2.1 h₂),
    fun hD => (hπ.3 (le_iff_imp.1 h.2.2 hD)).trans hc,
    fun hD => (hπ.4 (le_iff_imp.1 h.2.2 hD)).imp fun _ hπ => ⟨hπ.1, hπ.2.trans hc⟩⟩


variable (I) in
@[mono]
theorem MemBaseSet.mono (h : l₁ ≤ l₂) (hc : c₁ ≤ c₂)
    (hr : ∀ x ∈ Box.Icc I, r₁ x ≤ r₂ x) (hπ : l₁.MemBaseSet I c₁ r₁ π) : l₂.MemBaseSet I c₂ r₂ π :=
  hπ.mono' I h hc fun J _ => hr _ <| π.tag_mem_Icc J


theorem MemBaseSet.exists_common_compl
    (h₁ : l.MemBaseSet I c₁ r₁ π₁) (h₂ : l.MemBaseSet I c₂ r₂ π₂)
    (hU : π₁.iUnion = π₂.iUnion) :
    ∃ π : Prepartition I, π.iUnion = ↑I \ π₁.iUnion ∧
      (l.bDistortion → π.distortion ≤ c₁) ∧ (l.bDistortion → π.distortion ≤ c₂) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    c₁ c₂ : NNReal
    l : BoxIntegral.IntegrationParams
    r₁ r₂ : (ι → Real) → ↑(Set.Ioi 0)
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ r₁ π₁
    h₂ : l.MemBaseSet I c₂ r₂ π₂
    hU : Eq π₁.iUnion π₂.iUnion
    ⊢ Exists fun π => And (Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)) (And (Eq l.bD …
  -/
  wlog hc : c₁ ≤ c₂ with H
  · simpa [hU, _root_.and_comm] using
      @H _ _ I c₂ c₁ l r₂ r₁ π₂ π₁ h₂ h₁ hU.symm (le_of_not_le hc)
  /-
    ι✝ : Type u_1
    inst✝¹ : Fintype ι✝
    I✝ : BoxIntegral.Box ι✝
    c₁✝ c₂✝ : NNReal
    l✝ : BoxIntegral.IntegrationParams
    r₁✝ r₂✝ : (ι✝ → Real) → ↑(Set.Ioi 0)
    π₁✝ π₂✝ : BoxIntegral.TaggedPrepartition I✝
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    c₁ c₂ : NNReal
    l : BoxIntegral.IntegrationParams
    r₁ r₂ : (ι → Real) → ↑(Set.Ioi 0)
    π₁ π₂ : BoxIntegral.TaggedPrepartition I
    h₁ : l.MemBaseSet I c₁ r₁ π₁
    h₂ : l.MemBaseSet I c₂ r₂ π₂
    hU : Eq π₁.iUnion π₂.iUnion
    hc : LE.le c₁ c₂
    ⊢ Exists fun π => And (Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)) (And (Eq l.bD …
  -/
  by_cases hD : (l.bDistortion : Prop)
    /-
      case pos
      ι✝ : Type u_1
      inst✝¹ : Fintype ι✝
      I✝ : BoxIntegral.Box ι✝
      c₁✝ c₂✝ : NNReal
      l✝ : BoxIntegral.IntegrationParams
      r₁✝ r₂✝ : (ι✝ → Real) → ↑(Set.Ioi 0)
      π₁✝ π₂✝ : BoxIntegral.TaggedPrepartition I✝
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c₁ c₂ : NNReal
      l : BoxIntegral.IntegrationParams
      r₁ r₂ : (ι → Real) → ↑(Set.Ioi 0)
      π₁ π₂ : BoxIntegral.TaggedPrepartition I
      h₁ : l.MemBaseSet I c₁ r₁ π₁
      h₂ : l.MemBaseSet I c₂ r₂ π₂
      hU : Eq π₁.iUnion π₂.iUnion
      hc : LE.le c₁ c₂
      hD : Eq l.bDistortion Bool.true
      ⊢ Exists fun π => And (Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)) (And (Eq l.bD …
    -/
  · rcases h₁.4 hD with ⟨π, hπU, hπc⟩
    /-
      case pos.intro.intro
      ι✝ : Type u_1
      inst✝¹ : Fintype ι✝
      I✝ : BoxIntegral.Box ι✝
      c₁✝ c₂✝ : NNReal
      l✝ : BoxIntegral.IntegrationParams
      r₁✝ r₂✝ : (ι✝ → Real) → ↑(Set.Ioi 0)
      π₁✝ π₂✝ : BoxIntegral.TaggedPrepartition I✝
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c₁ c₂ : NNReal
      l : BoxIntegral.IntegrationParams
      r₁ r₂ : (ι → Real) → ↑(Set.Ioi 0)
      π₁ π₂ : BoxIntegral.TaggedPrepartition I
      h₁ : l.MemBaseSet I c₁ r₁ π₁
      h₂ : l.MemBaseSet I c₂ r₂ π₂
      hU : Eq π₁.iUnion π₂.iUnion
      hc : LE.le c₁ c₂
      hD : Eq l.bDistortion Bool.true
      π : BoxIntegral.Prepartition I
      hπU : Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
      hπc : LE.le π.distortion c₁
      ⊢ Exists fun π => And (Eq π.iUnion (SDiff.sdiff (↑I) π₁.iUnion)) (And (Eq l.bD …
    -/
    exact ⟨π, hπU, fun _ => hπc, fun _ => hπc.trans hc⟩
    /-
      🎉 no goals
    -/
  · exact ⟨π₁.toPrepartition.compl, π₁.toPrepartition.iUnion_compl,
      fun h => (hD h).elim, fun h => (hD h).elim⟩


protected theorem MemBaseSet.unionComplToSubordinate (hπ₁ : l.MemBaseSet I c r₁ π₁)
    (hle : ∀ x ∈ Box.Icc I, r₂ x ≤ r₁ x) {π₂ : Prepartition I} (hU : π₂.iUnion = ↑I \ π₁.iUnion)
    (hc : l.bDistortion → π₂.distortion ≤ c) :
    l.MemBaseSet I c r₁ (π₁.unionComplToSubordinate π₂ hU r₂) :=
  ⟨hπ₁.1.disjUnion ((π₂.isSubordinate_toSubordinate r₂).mono hle) _,
    fun h => (hπ₁.2 h).disjUnion (π₂.isHenstock_toSubordinate _) _,
    fun h => (distortion_unionComplToSubordinate _ _ _ _).trans_le (max_le (hπ₁.3 h) (hc h)),
                    /-
                      ι : Type u_1
                      inst✝ : Fintype ι
                      I : BoxIntegral.Box ι
                      c : NNReal
                      l : BoxIntegral.IntegrationParams
                      r₁ r₂ : (ι → Real) → ↑(Set.Ioi 0)
                      π₁ : BoxIntegral.TaggedPrepartition I
                      hπ₁ : l.MemBaseSet I c r₁ π₁
                      hle : ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) x → LE.le (r₂ x …
                      π₂ : BoxIntegral.Prepartition I
                      hU : Eq π₂.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
                      hc : Eq l.bDistortion Bool.true → LE.le π₂.distortion c
                      x✝ : Eq l.bDistortion Bool.true
                      ⊢ And (Eq Bot.bot.iUnion (SDiff.sdiff (↑I) (π₁.unionComplToSubordinate π₂ hU r …
                    -/
    fun _ => ⟨⊥, by simp⟩⟩
                    /-
                      🎉 no goals
                    -/


protected theorem MemBaseSet.filter (hπ : l.MemBaseSet I c r π) (p : Box ι → Prop) :
    l.MemBaseSet I c r (π.filter p) := by
  classical
  refine ⟨fun J hJ => hπ.1 J (π.mem_filter.1 hJ).1, fun hH J hJ => hπ.2 hH J (π.mem_filter.1 hJ).1,
    fun hD => (distortion_filter_le _ _).trans (hπ.3 hD), fun hD => ?_⟩
  rcases hπ.4 hD with ⟨π₁, hπ₁U, hc⟩
  set π₂ := π.filter fun J => ¬p J
  have : Disjoint π₁.iUnion π₂.iUnion := by
    simpa [π₂, hπ₁U] using disjoint_sdiff_self_left.mono_right sdiff_le
  refine ⟨π₁.disjUnion π₂.toPrepartition this, ?_, ?_⟩
  · suffices ↑I \ π.iUnion ∪ π.iUnion \ (π.filter p).iUnion = ↑I \ (π.filter p).iUnion by
      simp [π₂, *]
    have h : (π.filter p).iUnion ⊆ π.iUnion :=
      biUnion_subset_biUnion_left (Finset.filter_subset _ _)
    ext x
    fconstructor
    · rintro (⟨hxI, hxπ⟩ | ⟨hxπ, hxp⟩)
      exacts [⟨hxI, mt (@h x) hxπ⟩, ⟨π.iUnion_subset hxπ, hxp⟩]
    · rintro ⟨hxI, hxp⟩
      by_cases hxπ : x ∈ π.iUnion
      exacts [Or.inr ⟨hxπ, hxp⟩, Or.inl ⟨hxI, hxπ⟩]
  · have : (π.filter fun J => ¬p J).distortion ≤ c := (distortion_filter_le _ _).trans (hπ.3 hD)
    simpa [hc]


theorem biUnionTagged_memBaseSet {π : Prepartition I} {πi : ∀ J, TaggedPrepartition J}
    (h : ∀ J ∈ π, l.MemBaseSet J c r (πi J)) (hp : ∀ J ∈ π, (πi J).IsPartition)
    (hc : l.bDistortion → π.compl.distortion ≤ c) : l.MemBaseSet I c r (π.biUnionTagged πi) := by
  refine ⟨TaggedPrepartition.isSubordinate_biUnionTagged.2 fun J hJ => (h J hJ).1,
    fun hH => TaggedPrepartition.isHenstock_biUnionTagged.2 fun J hJ => (h J hJ).2 hH,
    fun hD => ?_, fun hD => ?_⟩
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c : NNReal
      l : BoxIntegral.IntegrationParams
      r : (ι → Real) → ↑(Set.Ioi 0)
      π : BoxIntegral.Prepartition I
      πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
      h : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → l.MemBaseSet J c r (πi J)
      hp : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → (πi J).IsPartition
      hc : Eq l.bDistortion Bool.true → LE.le π.compl.distortion c
      hD : Eq l.bDistortion Bool.true
      ⊢ LE.le (π.biUnionTagged πi).distortion c
    -/
  · rw [Prepartition.distortion_biUnionTagged, Finset.sup_le_iff]
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c : NNReal
      l : BoxIntegral.IntegrationParams
      r : (ι → Real) → ↑(Set.Ioi 0)
      π : BoxIntegral.Prepartition I
      πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
      h : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → l.MemBaseSet J c r (πi J)
      hp : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → (πi J).IsPartition
      hc : Eq l.bDistortion Bool.true → LE.le π.compl.distortion c
      hD : Eq l.bDistortion Bool.true
      ⊢ ∀ (b : BoxIntegral.Box ι), Membership.mem π.boxes b → LE.le (πi b).distortio …
    -/
    exact fun J hJ => (h J hJ).3 hD
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c : NNReal
      l : BoxIntegral.IntegrationParams
      r : (ι → Real) → ↑(Set.Ioi 0)
      π : BoxIntegral.Prepartition I
      πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
      h : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → l.MemBaseSet J c r (πi J)
      hp : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → (πi J).IsPartition
      hc : Eq l.bDistortion Bool.true → LE.le π.compl.distortion c
      hD : Eq l.bDistortion Bool.true
      ⊢ Exists fun π' => And (Eq π'.iUnion (SDiff.sdiff (↑I) (π.biUnionTagged πi).iU …
    -/
  · refine ⟨_, ?_, hc hD⟩
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c : NNReal
      l : BoxIntegral.IntegrationParams
      r : (ι → Real) → ↑(Set.Ioi 0)
      π : BoxIntegral.Prepartition I
      πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
      h : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → l.MemBaseSet J c r (πi J)
      hp : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → (πi J).IsPartition
      hc : Eq l.bDistortion Bool.true → LE.le π.compl.distortion c
      hD : Eq l.bDistortion Bool.true
      ⊢ Eq π.compl.iUnion (SDiff.sdiff (↑I) (π.biUnionTagged πi).iUnion)
    -/
    rw [π.iUnion_compl, ← π.iUnion_biUnion_partition hp]
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      c : NNReal
      l : BoxIntegral.IntegrationParams
      r : (ι → Real) → ↑(Set.Ioi 0)
      π : BoxIntegral.Prepartition I
      πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
      h : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → l.MemBaseSet J c r (πi J)
      hp : ∀ (J : BoxIntegral.Box ι), Membership.mem π J → (πi J).IsPartition
      hc : Eq l.bDistortion Bool.true → LE.le π.compl.distortion c
      hD : Eq l.bDistortion Bool.true
      ⊢ Eq (SDiff.sdiff (↑I) (π.biUnion fun J => (πi J).toPrepartition).iUnion) (SDi …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[mono]
theorem RCond.mono {ι : Type*} {r : (ι → ℝ) → Ioi (0 : ℝ)} (h : l₁ ≤ l₂) (hr : l₂.RCond r) :
    l₁.RCond r :=
  fun hR => hr (le_iff_imp.1 h.1 hR)


nonrec theorem RCond.min {ι : Type*} {r₁ r₂ : (ι → ℝ) → Ioi (0 : ℝ)} (h₁ : l.RCond r₁)
    (h₂ : l.RCond r₂) : l.RCond fun x => min (r₁ x) (r₂ x) :=
  fun hR x => congr_arg₂ min (h₁ hR x) (h₂ hR x)


@[gcongr, mono]
theorem toFilterDistortion_mono (I : Box ι) (h : l₁ ≤ l₂) (hc : c₁ ≤ c₂) :
    l₁.toFilterDistortion I c₁ ≤ l₂.toFilterDistortion I c₂ :=
  iInf_mono fun _ =>
    iInf_mono' fun hr =>
      ⟨hr.mono h, principal_mono.2 fun _ => MemBaseSet.mono I h hc fun _ _ => le_rfl⟩


@[gcongr, mono]
theorem toFilter_mono (I : Box ι) {l₁ l₂ : IntegrationParams} (h : l₁ ≤ l₂) :
    l₁.toFilter I ≤ l₂.toFilter I :=
  iSup_mono fun _ => toFilterDistortion_mono I h le_rfl


@[gcongr, mono]
theorem toFilteriUnion_mono (I : Box ι) {l₁ l₂ : IntegrationParams} (h : l₁ ≤ l₂)
    (π₀ : Prepartition I) : l₁.toFilteriUnion I π₀ ≤ l₂.toFilteriUnion I π₀ :=
  iSup_mono fun _ => inf_le_inf_right _ <| toFilterDistortion_mono _ h le_rfl


theorem toFilteriUnion_congr (I : Box ι) (l : IntegrationParams) {π₁ π₂ : Prepartition I}
    (h : π₁.iUnion = π₂.iUnion) : l.toFilteriUnion I π₁ = l.toFilteriUnion I π₂ := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    ⊢ Eq (BoxIntegral.IntegrationParams.toFilteriUnion I π₁) (BoxIntegral.Integrat …
  -/
  simp only [toFilteriUnion, toFilterDistortioniUnion, h]
  /-
    🎉 no goals
  -/


theorem hasBasis_toFilterDistortion (l : IntegrationParams) (I : Box ι) (c : ℝ≥0) :
    (l.toFilterDistortion I c).HasBasis l.RCond fun r => { π | l.MemBaseSet I c r π } :=
  hasBasis_biInf_principal'
    (fun _ hr₁ _ hr₂ =>
      ⟨_, hr₁.min hr₂, fun _ => MemBaseSet.mono _ le_rfl le_rfl fun _ _ => min_le_left _ _,
        fun _ => MemBaseSet.mono _ le_rfl le_rfl fun _ _ => min_le_right _ _⟩)
    ⟨fun _ => ⟨1, Set.mem_Ioi.2 zero_lt_one⟩, fun _ _ => rfl⟩


theorem hasBasis_toFilterDistortioniUnion (l : IntegrationParams) (I : Box ι) (c : ℝ≥0)
    (π₀ : Prepartition I) :
    (l.toFilterDistortioniUnion I c π₀).HasBasis l.RCond fun r =>
      { π | l.MemBaseSet I c r π ∧ π.iUnion = π₀.iUnion } :=
  (l.hasBasis_toFilterDistortion I c).inf_principal _


theorem hasBasis_toFilteriUnion (l : IntegrationParams) (I : Box ι) (π₀ : Prepartition I) :
    (l.toFilteriUnion I π₀).HasBasis (fun r : ℝ≥0 → (ι → ℝ) → Ioi (0 : ℝ) => ∀ c, l.RCond (r c))
      fun r => { π | ∃ c, l.MemBaseSet I c (r c) π ∧ π.iUnion = π₀.iUnion } := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    π₀ : BoxIntegral.Prepartition I
    ⊢ (BoxIntegral.IntegrationParams.toFilteriUnion I π₀).HasBasis (fun r => ∀ (c  …
  -/
  have := fun c => l.hasBasis_toFilterDistortioniUnion I c π₀
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    π₀ : BoxIntegral.Prepartition I
    this : ∀ (c : NNReal), (l.toFilterDistortioniUnion I c π₀).HasBasis l.RCond fu …
    ⊢ (BoxIntegral.IntegrationParams.toFilteriUnion I π₀).HasBasis (fun r => ∀ (c  …
  -/
  simpa only [setOf_and, setOf_exists] using hasBasis_iSup this
  /-
    🎉 no goals
  -/


theorem hasBasis_toFilteriUnion_top (l : IntegrationParams) (I : Box ι) :
    (l.toFilteriUnion I ⊤).HasBasis (fun r : ℝ≥0 → (ι → ℝ) → Ioi (0 : ℝ) => ∀ c, l.RCond (r c))
      fun r => { π | ∃ c, l.MemBaseSet I c (r c) π ∧ π.IsPartition } := by
  simpa only [TaggedPrepartition.isPartition_iff_iUnion_eq, Prepartition.iUnion_top] using
    l.hasBasis_toFilteriUnion I ⊤


theorem hasBasis_toFilter (l : IntegrationParams) (I : Box ι) :
    (l.toFilter I).HasBasis (fun r : ℝ≥0 → (ι → ℝ) → Ioi (0 : ℝ) => ∀ c, l.RCond (r c))
      fun r => { π | ∃ c, l.MemBaseSet I c (r c) π } := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    ⊢ (l.toFilter I).HasBasis (fun r => ∀ (c : NNReal), l.RCond (r c)) fun r => se …
  -/
  simpa only [setOf_exists] using hasBasis_iSup (l.hasBasis_toFilterDistortion I)
  /-
    🎉 no goals
  -/


theorem tendsto_embedBox_toFilteriUnion_top (l : IntegrationParams) (h : I ≤ J) :
    Tendsto (TaggedPrepartition.embedBox I J h) (l.toFilteriUnion I ⊤)
      (l.toFilteriUnion J (Prepartition.single J I h)) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I J : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    h : LE.le I J
    ⊢ Filter.Tendsto (⇑(BoxIntegral.TaggedPrepartition.embedBox I J h)) (BoxIntegr …
  -/
  simp only [toFilteriUnion, tendsto_iSup]; intro c
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I J : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    h : LE.le I J
    c : NNReal
    ⊢ Filter.Tendsto (⇑(BoxIntegral.TaggedPrepartition.embedBox I J h)) (l.toFilte …
  -/
  set π₀ := Prepartition.single J I h
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I J : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    h : LE.le I J
    c : NNReal
    π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
    ⊢ Filter.Tendsto (⇑(BoxIntegral.TaggedPrepartition.embedBox I J h)) (l.toFilte …
  -/
  refine le_iSup_of_le (max c π₀.compl.distortion) ?_
  refine ((l.hasBasis_toFilterDistortioniUnion I c ⊤).tendsto_iff
    (l.hasBasis_toFilterDistortioniUnion J _ _)).2 fun r hr => ?_
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I J : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    h : LE.le I J
    c : NNReal
    π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
    r : (ι → Real) → ↑(Set.Ioi 0)
    hr : l.RCond r
    ⊢ Exists fun ia => And (l.RCond ia) (∀ (x : BoxIntegral.TaggedPrepartition I), …
  -/
  refine ⟨r, hr, fun π hπ => ?_⟩
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I J : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    h : LE.le I J
    c : NNReal
    π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
    r : (ι → Real) → ↑(Set.Ioi 0)
    hr : l.RCond r
    π : BoxIntegral.TaggedPrepartition I
    hπ : Membership.mem (setOf fun π => And (l.MemBaseSet I c r π) (Eq π.iUnion To …
    ⊢ Membership.mem (setOf fun π => And (l.MemBaseSet J (Max.max c π₀.compl.disto …
  -/
  rw [mem_setOf_eq, Prepartition.iUnion_top] at hπ
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I J : BoxIntegral.Box ι
    l : BoxIntegral.IntegrationParams
    h : LE.le I J
    c : NNReal
    π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
    r : (ι → Real) → ↑(Set.Ioi 0)
    hr : l.RCond r
    π : BoxIntegral.TaggedPrepartition I
    hπ : And (l.MemBaseSet I c r π) (Eq π.iUnion ↑I)
    ⊢ Membership.mem (setOf fun π => And (l.MemBaseSet J (Max.max c π₀.compl.disto …
  -/
  refine ⟨⟨hπ.1.1, hπ.1.2, fun hD => le_trans (hπ.1.3 hD) (le_max_left _ _), fun _ => ?_⟩, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I J : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      h : LE.le I J
      c : NNReal
      π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
      r : (ι → Real) → ↑(Set.Ioi 0)
      hr : l.RCond r
      π : BoxIntegral.TaggedPrepartition I
      hπ : And (l.MemBaseSet I c r π) (Eq π.iUnion ↑I)
      x✝ : Eq l.bDistortion Bool.true
      ⊢ Exists fun π' => And (Eq π'.iUnion (SDiff.sdiff (↑J) ((BoxIntegral.TaggedPre …
    -/
  · refine ⟨_, π₀.iUnion_compl.trans ?_, le_max_right _ _⟩
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I J : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      h : LE.le I J
      c : NNReal
      π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
      r : (ι → Real) → ↑(Set.Ioi 0)
      hr : l.RCond r
      π : BoxIntegral.TaggedPrepartition I
      hπ : And (l.MemBaseSet I c r π) (Eq π.iUnion ↑I)
      x✝ : Eq l.bDistortion Bool.true
      ⊢ Eq (SDiff.sdiff (↑J) π₀.iUnion) (SDiff.sdiff (↑J) ((BoxIntegral.TaggedPrepar …
    -/
    congr 1
    /-
      case refine_1.e_a
      ι : Type u_1
      inst✝ : Fintype ι
      I J : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      h : LE.le I J
      c : NNReal
      π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
      r : (ι → Real) → ↑(Set.Ioi 0)
      hr : l.RCond r
      π : BoxIntegral.TaggedPrepartition I
      hπ : And (l.MemBaseSet I c r π) (Eq π.iUnion ↑I)
      x✝ : Eq l.bDistortion Bool.true
      ⊢ Eq π₀.iUnion ((BoxIntegral.TaggedPrepartition.embedBox I J h) π).iUnion
    -/
    exact (Prepartition.iUnion_single h).trans hπ.2.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I J : BoxIntegral.Box ι
      l : BoxIntegral.IntegrationParams
      h : LE.le I J
      c : NNReal
      π₀ : BoxIntegral.Prepartition J := BoxIntegral.Prepartition.single J I h
      r : (ι → Real) → ↑(Set.Ioi 0)
      hr : l.RCond r
      π : BoxIntegral.TaggedPrepartition I
      hπ : And (l.MemBaseSet I c r π) (Eq π.iUnion ↑I)
      ⊢ Eq ((BoxIntegral.TaggedPrepartition.embedBox I J h) π).iUnion π₀.iUnion
    -/
  · exact hπ.2.trans (Prepartition.iUnion_single _).symm
    /-
      🎉 no goals
    -/


theorem exists_memBaseSet_le_iUnion_eq (l : IntegrationParams) (π₀ : Prepartition I)
    (hc₁ : π₀.distortion ≤ c) (hc₂ : π₀.compl.distortion ≤ c) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    ∃ π, l.MemBaseSet I c r π ∧ π.toPrepartition ≤ π₀ ∧ π.iUnion = π₀.iUnion := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    c : NNReal
    l : BoxIntegral.IntegrationParams
    π₀ : BoxIntegral.Prepartition I
    hc₁ : LE.le π₀.distortion c
    hc₂ : LE.le π₀.compl.distortion c
    r : (ι → Real) → ↑(Set.Ioi 0)
    ⊢ Exists fun π => And (l.MemBaseSet I c r π) (And (LE.le π.toPrepartition π₀)  …
  -/
  rcases π₀.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r with ⟨π, hle, hH, hr, hd, hU⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    c : NNReal
    l : BoxIntegral.IntegrationParams
    π₀ : BoxIntegral.Prepartition I
    hc₁ : LE.le π₀.distortion c
    hc₂ : LE.le π₀.compl.distortion c
    r : (ι → Real) → ↑(Set.Ioi 0)
    π : BoxIntegral.TaggedPrepartition I
    hle : LE.le π.toPrepartition π₀
    hH : π.IsHenstock
    hr : π.IsSubordinate r
    hd : Eq π.distortion π₀.distortion
    hU : Eq π.iUnion π₀.iUnion
    ⊢ Exists fun π => And (l.MemBaseSet I c r π) (And (LE.le π.toPrepartition π₀)  …
  -/
  refine ⟨π, ⟨hr, fun _ => hH, fun _ => hd.trans_le hc₁, fun _ => ⟨π₀.compl, ?_, hc₂⟩⟩, ⟨hle, hU⟩⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    c : NNReal
    l : BoxIntegral.IntegrationParams
    π₀ : BoxIntegral.Prepartition I
    hc₁ : LE.le π₀.distortion c
    hc₂ : LE.le π₀.compl.distortion c
    r : (ι → Real) → ↑(Set.Ioi 0)
    π : BoxIntegral.TaggedPrepartition I
    hle : LE.le π.toPrepartition π₀
    hH : π.IsHenstock
    hr : π.IsSubordinate r
    hd : Eq π.distortion π₀.distortion
    hU : Eq π.iUnion π₀.iUnion
    x✝ : Eq l.bDistortion Bool.true
    ⊢ Eq π₀.compl.iUnion (SDiff.sdiff (↑I) π.iUnion)
  -/
  exact Prepartition.compl_congr hU ▸ π.toPrepartition.iUnion_compl
  /-
    🎉 no goals
  -/


theorem exists_memBaseSet_isPartition (l : IntegrationParams) (I : Box ι) (hc : I.distortion ≤ c)
    (r : (ι → ℝ) → Ioi (0 : ℝ)) : ∃ π, l.MemBaseSet I c r π ∧ π.IsPartition := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    c : NNReal
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    hc : LE.le I.distortion c
    r : (ι → Real) → ↑(Set.Ioi 0)
    ⊢ Exists fun π => And (l.MemBaseSet I c r π) π.IsPartition
  -/
  rw [← Prepartition.distortion_top] at hc
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    c : NNReal
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    hc : LE.le Top.top.distortion c
    r : (ι → Real) → ↑(Set.Ioi 0)
    ⊢ Exists fun π => And (l.MemBaseSet I c r π) π.IsPartition
  -/
  have hc' : (⊤ : Prepartition I).compl.distortion ≤ c := by simp
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    c : NNReal
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    hc : LE.le Top.top.distortion c
    r : (ι → Real) → ↑(Set.Ioi 0)
    hc' : LE.le Top.top.compl.distortion c
    ⊢ Exists fun π => And (l.MemBaseSet I c r π) π.IsPartition
  -/
  simpa [isPartition_iff_iUnion_eq] using l.exists_memBaseSet_le_iUnion_eq ⊤ hc hc' r
  /-
    🎉 no goals
  -/


theorem toFilterDistortioniUnion_neBot (l : IntegrationParams) (I : Box ι) (π₀ : Prepartition I)
    (hc₁ : π₀.distortion ≤ c) (hc₂ : π₀.compl.distortion ≤ c) :
    (l.toFilterDistortioniUnion I c π₀).NeBot :=
  ((l.hasBasis_toFilterDistortion I _).inf_principal _).neBot_iff.2
    fun {r} _ => (l.exists_memBaseSet_le_iUnion_eq π₀ hc₁ hc₂ r).imp fun _ hπ => ⟨hπ.1, hπ.2.2⟩


instance toFilterDistortioniUnion_neBot' (l : IntegrationParams) (I : Box ι) (π₀ : Prepartition I) :
    (l.toFilterDistortioniUnion I (max π₀.distortion π₀.compl.distortion) π₀).NeBot :=
  l.toFilterDistortioniUnion_neBot I π₀ (le_max_left _ _) (le_max_right _ _)


instance toFilterDistortion_neBot (l : IntegrationParams) (I : Box ι) :
    (l.toFilterDistortion I I.distortion).NeBot := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I✝ J : BoxIntegral.Box ι
    c c₁ c₂ : NNReal
    l✝ l₁ l₂ : BoxIntegral.IntegrationParams
    r₁ r₂ : (ι → Real) → ↑(Set.Ioi 0)
    π π₁ π₂ : BoxIntegral.TaggedPrepartition I✝
    r : (ι → Real) → ↑(Set.Ioi 0)
    l : BoxIntegral.IntegrationParams
    I : BoxIntegral.Box ι
    ⊢ (l.toFilterDistortion I I.distortion).NeBot
  -/
  simpa using (l.toFilterDistortioniUnion_neBot' I ⊤).mono inf_le_left
  /-
    🎉 no goals
  -/


instance toFilter_neBot (l : IntegrationParams) (I : Box ι) : (l.toFilter I).NeBot :=
  (l.toFilterDistortion_neBot I).mono <| le_iSup _ _


instance toFilteriUnion_neBot (l : IntegrationParams) (I : Box ι) (π₀ : Prepartition I) :
    (l.toFilteriUnion I π₀).NeBot :=
  (l.toFilterDistortioniUnion_neBot' I π₀).mono <|
    le_iSup (fun c => l.toFilterDistortioniUnion I c π₀) _


theorem eventually_isPartition (l : IntegrationParams) (I : Box ι) :
    ∀ᶠ π in l.toFilteriUnion I ⊤, TaggedPrepartition.IsPartition π :=
  eventually_iSup.2 fun _ =>
    eventually_inf_principal.2 <|
      Eventually.of_forall fun π h =>
        π.isPartition_iff_iUnion_eq.2 (h.trans Prepartition.iUnion_top)


