/-- A function on `Box ι` is called box additive if for every box `J` and a partition `π` of `J`
we have `f J = ∑ Ji ∈ π.boxes, f Ji`. A function is called box additive on subboxes of `I : Box ι`
if the same property holds for `J ≤ I`. We formalize these two notions in the same definition
using `I : WithBot (Box ι)`: the value `I = ⊤` corresponds to functions box additive on the whole
space. -/
structure BoxAdditiveMap (ι M : Type*) [AddCommMonoid M] (I : WithTop (Box ι)) where
  /-- The function underlying this additive map. -/
  toFun : Box ι → M
  sum_partition_boxes' : ∀ J : Box ι, ↑J ≤ I → ∀ π : Prepartition J, π.IsPartition →
    ∑ Ji ∈ π.boxes, toFun Ji = toFun J



/-- A function on `Box ι` is called box additive if for every box `J` and a partition `π` of `J`
we have `f J = ∑ Ji ∈ π.boxes, f Ji`. -/
scoped notation:25 ι " →ᵇᵃ " M => BoxIntegral.BoxAdditiveMap ι M ⊤


@[inherit_doc] scoped notation:25 ι " →ᵇᵃ[" I "] " M => BoxIntegral.BoxAdditiveMap ι M I


instance : FunLike (ι →ᵇᵃ[I₀] M) (Box ι) M where
  coe := toFun
                             /-
                               ι : Type u_1
                               M : Type u_2
                               n : Nat
                               N : Type u_3
                               inst✝¹ : AddCommMonoid M
                               inst✝ : AddCommMonoid N
                               I₀ : WithTop (BoxIntegral.Box ι)
                               I : BoxIntegral.Box ι
                               i : ι
                               f g : BoxIntegral.BoxAdditiveMap ι M I₀
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem coe_mk (f h) : ⇑(mk f h : ι →ᵇᵃ[I₀] M) = f := rfl


theorem coe_injective : Injective fun (f : ι →ᵇᵃ[I₀] M) x => f x :=
  DFunLike.coe_injective


theorem coe_inj {f g : ι →ᵇᵃ[I₀] M} : (f : Box ι → M) = g ↔ f = g := DFunLike.coe_fn_eq


theorem sum_partition_boxes (f : ι →ᵇᵃ[I₀] M) (hI : ↑I ≤ I₀) {π : Prepartition I}
    (h : π.IsPartition) : ∑ J ∈ π.boxes, f J = f I :=
  f.sum_partition_boxes' I hI π h


@[simps (config := .asFn)]
instance : Zero (ι →ᵇᵃ[I₀] M) :=
  ⟨⟨0, fun _ _ _ _ => sum_const_zero⟩⟩


instance : Inhabited (ι →ᵇᵃ[I₀] M) :=
  ⟨0⟩


instance : Add (ι →ᵇᵃ[I₀] M) :=
  ⟨fun f g =>
    ⟨f + g, fun I hI π hπ => by
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        N : Type u_3
        inst✝¹ : AddCommMonoid M
        inst✝ : AddCommMonoid N
        I₀ : WithTop (BoxIntegral.Box ι)
        I✝ : BoxIntegral.Box ι
        i : ι
        f g : BoxIntegral.BoxAdditiveMap ι M I₀
        I : BoxIntegral.Box ι
        hI : LE.le (↑I) I₀
        π : BoxIntegral.Prepartition I
        hπ : π.IsPartition
        ⊢ Eq (π.boxes.sum fun Ji => HAdd.hAdd (⇑f) (⇑g) Ji) (HAdd.hAdd (⇑f) (⇑g) I)
      -/
      simp only [Pi.add_apply, sum_add_distrib, sum_partition_boxes _ hI hπ]⟩⟩
      /-
        🎉 no goals
      -/


instance {R} [Monoid R] [DistribMulAction R M] : SMul R (ι →ᵇᵃ[I₀] M) :=
  ⟨fun r f =>
    ⟨r • (f : Box ι → M), fun I hI π hπ => by
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        N : Type u_3
        inst✝³ : AddCommMonoid M
        inst✝² : AddCommMonoid N
        I₀ : WithTop (BoxIntegral.Box ι)
        I✝ : BoxIntegral.Box ι
        i : ι
        R : Type ?u.9686
        inst✝¹ : Monoid R
        inst✝ : DistribMulAction R M
        r : R
        f : BoxIntegral.BoxAdditiveMap ι M I₀
        I : BoxIntegral.Box ι
        hI : LE.le (↑I) I₀
        π : BoxIntegral.Prepartition I
        hπ : π.IsPartition
        ⊢ Eq (π.boxes.sum fun Ji => HSMul.hSMul r (⇑f) Ji) (HSMul.hSMul r (⇑f) I)
      -/
      simp only [Pi.smul_apply, ← smul_sum, sum_partition_boxes _ hI hπ]⟩⟩
      /-
        🎉 no goals
      -/


instance : AddCommMonoid (ι →ᵇᵃ[I₀] M) :=
  Function.Injective.addCommMonoid _ coe_injective rfl (fun _ _ => rfl) fun _ _ => rfl


@[simp]
theorem map_split_add (f : ι →ᵇᵃ[I₀] M) (hI : ↑I ≤ I₀) (i : ι) (x : ℝ) :
    (I.splitLower i x).elim' 0 f + (I.splitUpper i x).elim' 0 f = f I := by
  /-
    ι : Type u_1
    M : Type u_2
    inst✝ : AddCommMonoid M
    I₀ : WithTop (BoxIntegral.Box ι)
    I : BoxIntegral.Box ι
    f : BoxIntegral.BoxAdditiveMap ι M I₀
    hI : LE.le (↑I) I₀
    i : ι
    x : Real
    ⊢ Eq (HAdd.hAdd (Option.elim' 0 (⇑f) (I.splitLower i x)) (Option.elim' 0 (⇑f)  …
  -/
  rw [← f.sum_partition_boxes hI (isPartitionSplit I i x), sum_split_boxes]
  /-
    🎉 no goals
  -/


/-- If `f` is box-additive on subboxes of `I₀`, then it is box-additive on subboxes of any
`I ≤ I₀`. -/
@[simps]
def restrict (f : ι →ᵇᵃ[I₀] M) (I : WithTop (Box ι)) (hI : I ≤ I₀) : ι →ᵇᵃ[I] M :=
  ⟨f, fun J hJ => f.2 J (hJ.trans hI)⟩


/-- If `f : Box ι → M` is box additive on partitions of the form `split I i x`, then it is box
additive. -/
def ofMapSplitAdd [Finite ι] (f : Box ι → M) (I₀ : WithTop (Box ι))
    (hf : ∀ I : Box ι, ↑I ≤ I₀ → ∀ {i x}, x ∈ Ioo (I.lower i) (I.upper i) →
      (I.splitLower i x).elim' 0 f + (I.splitUpper i x).elim' 0 f = f I) :
    ι →ᵇᵃ[I₀] M := by
  classical
  refine ⟨f, ?_⟩
  replace hf : ∀ I : Box ι, ↑I ≤ I₀ → ∀ s, (∑ J ∈ (splitMany I s).boxes, f J) = f I := by
    intro I hI s
    induction' s using Finset.induction_on with a s _ ihs
    · simp
    rw [splitMany_insert, inf_split, ← ihs, biUnion_boxes, sum_biUnion_boxes]
    refine Finset.sum_congr rfl fun J' hJ' => ?_
    by_cases h : a.2 ∈ Ioo (J'.lower a.1) (J'.upper a.1)
    · rw [sum_split_boxes]
      exact hf _ ((WithTop.coe_le_coe.2 <| le_of_mem _ hJ').trans hI) h
    · rw [split_of_not_mem_Ioo h, top_boxes, Finset.sum_singleton]
  intro I hI π hπ
  have Hle : ∀ J ∈ π, ↑J ≤ I₀ := fun J hJ => (WithTop.coe_le_coe.2 <| π.le_of_mem hJ).trans hI
  rcases hπ.exists_splitMany_le with ⟨s, hs⟩
  rw [← hf _ hI, ← inf_of_le_right hs, inf_splitMany, biUnion_boxes, sum_biUnion_boxes]
  exact Finset.sum_congr rfl fun J hJ => (hf _ (Hle _ hJ) _).symm


/-- If `g : M → N` is an additive map and `f` is a box additive map, then `g ∘ f` is a box additive
map. -/
@[simps (config := .asFn)]
def map (f : ι →ᵇᵃ[I₀] M) (g : M →+ N) : ι →ᵇᵃ[I₀] N where
  toFun := g ∘ f
                                       /-
                                         ι : Type u_1
                                         M : Type u_2
                                         n : Nat
                                         N : Type u_3
                                         inst✝¹ : AddCommMonoid M
                                         inst✝ : AddCommMonoid N
                                         I₀ : WithTop (BoxIntegral.Box ι)
                                         I✝ : BoxIntegral.Box ι
                                         i : ι
                                         f : BoxIntegral.BoxAdditiveMap ι M I₀
                                         g : AddMonoidHom M N
                                         I : BoxIntegral.Box ι
                                         hI : LE.le (↑I) I₀
                                         π : BoxIntegral.Prepartition I
                                         hπ : π.IsPartition
                                         ⊢ Eq (π.boxes.sum fun Ji => Function.comp (⇑g) (⇑f) Ji) (Function.comp (⇑g) (⇑ …
                                       -/
  sum_partition_boxes' I hI π hπ := by simp_rw [comp, ← map_sum, f.sum_partition_boxes hI hπ]
                                       /-
                                         🎉 no goals
                                       -/


/-- If `f` is a box additive function on subboxes of `I` and `π₁`, `π₂` are two prepartitions of
`I` that cover the same part of `I`, then `∑ J ∈ π₁.boxes, f J = ∑ J ∈ π₂.boxes, f J`. -/
theorem sum_boxes_congr [Finite ι] (f : ι →ᵇᵃ[I₀] M) (hI : ↑I ≤ I₀) {π₁ π₂ : Prepartition I}
    (h : π₁.iUnion = π₂.iUnion) : ∑ J ∈ π₁.boxes, f J = ∑ J ∈ π₂.boxes, f J := by
  rcases exists_splitMany_inf_eq_filter_of_finite {π₁, π₂} ((finite_singleton _).insert _) with
    ⟨s, hs⟩
  /-
    case intro
    ι : Type u_1
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    I₀ : WithTop (BoxIntegral.Box ι)
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    f : BoxIntegral.BoxAdditiveMap ι M I₀
    hI : LE.le (↑I) I₀
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    s : Finset (Prod ι Real)
    hs : ∀ (π : BoxIntegral.Prepartition I), Membership.mem (Insert.insert π₁ (Sin …
    ⊢ Eq (π₁.boxes.sum fun J => f J) (π₂.boxes.sum fun J => f J)
  -/
  simp only [inf_splitMany] at hs
  /-
    case intro
    ι : Type u_1
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    I₀ : WithTop (BoxIntegral.Box ι)
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    f : BoxIntegral.BoxAdditiveMap ι M I₀
    hI : LE.le (↑I) I₀
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    s : Finset (Prod ι Real)
    hs : ∀ (π : BoxIntegral.Prepartition I), Membership.mem (Insert.insert π₁ (Sin …
    ⊢ Eq (π₁.boxes.sum fun J => f J) (π₂.boxes.sum fun J => f J)
  -/
  rcases hs _ (Or.inl rfl), hs _ (Or.inr rfl) with ⟨h₁, h₂⟩; clear hs
  /-
    case intro
    ι : Type u_1
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    I₀ : WithTop (BoxIntegral.Box ι)
    I : BoxIntegral.Box ι
    inst✝ : Finite ι
    f : BoxIntegral.BoxAdditiveMap ι M I₀
    hI : LE.le (↑I) I₀
    π₁ π₂ : BoxIntegral.Prepartition I
    h : Eq π₁.iUnion π₂.iUnion
    s : Finset (Prod ι Real)
    h₁ : Eq (π₁.biUnion fun J => BoxIntegral.Prepartition.splitMany J s) ((BoxInte …
    h₂ : Eq (π₂.biUnion fun J => BoxIntegral.Prepartition.splitMany J s) ((BoxInte …
    ⊢ Eq (π₁.boxes.sum fun J => f J) (π₂.boxes.sum fun J => f J)
  -/
  rw [h] at h₁
  calc
    ∑ J ∈ π₁.boxes, f J = ∑ J ∈ π₁.boxes, ∑ J' ∈ (splitMany J s).boxes, f J' :=
      Finset.sum_congr rfl fun J hJ => (f.sum_partition_boxes ?_ (isPartition_splitMany _ _)).symm
    _ = ∑ J ∈ (π₁.biUnion fun J => splitMany J s).boxes, f J := (sum_biUnion_boxes _ _ _).symm
    _ = ∑ J ∈ (π₂.biUnion fun J => splitMany J s).boxes, f J := by rw [h₁, h₂]
    _ = ∑ J ∈ π₂.boxes, ∑ J' ∈ (splitMany J s).boxes, f J' := sum_biUnion_boxes _ _ _
    _ = ∑ J ∈ π₂.boxes, f J :=
      Finset.sum_congr rfl fun J hJ => f.sum_partition_boxes ?_ (isPartition_splitMany _ _)
  exacts [(WithTop.coe_le_coe.2 <| π₁.le_of_mem hJ).trans hI,
    (WithTop.coe_le_coe.2 <| π₂.le_of_mem hJ).trans hI]


/-- If `f` is a box-additive map, then so is the map sending `I` to the scalar multiplication
by `f I` as a continuous linear map from `E` to itself. -/
def toSMul (f : ι →ᵇᵃ[I₀] ℝ) : ι →ᵇᵃ[I₀] E →L[ℝ] E :=
  f.map (ContinuousLinearMap.lsmul ℝ ℝ).toLinearMap.toAddMonoidHom


@[simp]
theorem toSMul_apply (f : ι →ᵇᵃ[I₀] ℝ) (I : Box ι) (x : E) : f.toSMul I x = f I • x := rfl


/-- Given a box `I₀` in `ℝⁿ⁺¹`, `f x : Box (Fin n) → G` is a family of functions indexed by a real
`x` and for `x ∈ [I₀.lower i, I₀.upper i]`, `f x` is box-additive on subboxes of the `i`-th face of
`I₀`, then `fun J ↦ f (J.upper i) (J.face i) - f (J.lower i) (J.face i)` is box-additive on subboxes
of `I₀`. -/
@[simps!]
def upperSubLower.{u} {G : Type u} [AddCommGroup G] (I₀ : Box (Fin (n + 1))) (i : Fin (n + 1))
    (f : ℝ → Box (Fin n) → G) (fb : Icc (I₀.lower i) (I₀.upper i) → Fin n →ᵇᵃ[I₀.face i] G)
    (hf : ∀ (x) (hx : x ∈ Icc (I₀.lower i) (I₀.upper i)) (J), f x J = fb ⟨x, hx⟩ J) :
    Fin (n + 1) →ᵇᵃ[I₀] G :=
  ofMapSplitAdd (fun J : Box (Fin (n + 1)) => f (J.upper i) (J.face i) - f (J.lower i) (J.face i))
    I₀
    (by
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        N : Type u_3
        inst✝² : AddCommMonoid M
        inst✝¹ : AddCommMonoid N
        I₀✝ : WithTop (BoxIntegral.Box ι)
        I : BoxIntegral.Box ι
        i✝ : ι
        G : Type u
        inst✝ : AddCommGroup G
        I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 1)
        f : Real → BoxIntegral.Box (Fin n) → G
        fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
        hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
        ⊢ ∀ (I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))), LE.le ↑I ↑I₀ → ∀ {i_1 : Fin ( …
      -/
      intro J hJ j x
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        N : Type u_3
        inst✝² : AddCommMonoid M
        inst✝¹ : AddCommMonoid N
        I₀✝ : WithTop (BoxIntegral.Box ι)
        I : BoxIntegral.Box ι
        i✝ : ι
        G : Type u
        inst✝ : AddCommGroup G
        I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 1)
        f : Real → BoxIntegral.Box (Fin n) → G
        fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
        hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJ : LE.le ↑J ↑I₀
        j : Fin (HAdd.hAdd n 1)
        x : Real
        ⊢ Membership.mem (Set.Ioo (J.lower j) (J.upper j)) x → Eq (HAdd.hAdd (Option.e …
      -/
      rw [WithTop.coe_le_coe] at hJ
      /-
        ι : Type u_1
        M : Type u_2
        n : Nat
        N : Type u_3
        inst✝² : AddCommMonoid M
        inst✝¹ : AddCommMonoid N
        I₀✝ : WithTop (BoxIntegral.Box ι)
        I : BoxIntegral.Box ι
        i✝ : ι
        G : Type u
        inst✝ : AddCommGroup G
        I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 1)
        f : Real → BoxIntegral.Box (Fin n) → G
        fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
        hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJ : LE.le J I₀
        j : Fin (HAdd.hAdd n 1)
        x : Real
        ⊢ Membership.mem (Set.Ioo (J.lower j) (J.upper j)) x → Eq (HAdd.hAdd (Option.e …
      -/
      refine i.succAboveCases (fun hx => ?_) (fun j hx => ?_) j
      · simp only [Box.splitLower_def hx, Box.splitUpper_def hx, update_self, ← WithBot.some_eq_coe,
          Option.elim', Box.face, Function.comp_def, update_of_ne (Fin.succAbove_ne _ _)]
        /-
          case refine_1
          ι : Type u_1
          M : Type u_2
          n : Nat
          N : Type u_3
          inst✝² : AddCommMonoid M
          inst✝¹ : AddCommMonoid N
          I₀✝ : WithTop (BoxIntegral.Box ι)
          I : BoxIntegral.Box ι
          i✝ : ι
          G : Type u
          inst✝ : AddCommGroup G
          I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 1)
          f : Real → BoxIntegral.Box (Fin n) → G
          fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
          hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
          J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          hJ : LE.le J I₀
          j : Fin (HAdd.hAdd n 1)
          x : Real
          hx : Membership.mem (Set.Ioo (J.lower i) (J.upper i)) x
          ⊢ Eq (HAdd.hAdd (HSub.hSub (f x { lower := fun x => J.lower (i.succAbove x), u …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · have : (J.face i : WithTop (Box (Fin n))) ≤ I₀.face i :=
          WithTop.coe_le_coe.2 (face_mono hJ i)
        /-
          case refine_2
          ι : Type u_1
          M : Type u_2
          n : Nat
          N : Type u_3
          inst✝² : AddCommMonoid M
          inst✝¹ : AddCommMonoid N
          I₀✝ : WithTop (BoxIntegral.Box ι)
          I : BoxIntegral.Box ι
          i✝ : ι
          G : Type u
          inst✝ : AddCommGroup G
          I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 1)
          f : Real → BoxIntegral.Box (Fin n) → G
          fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
          hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
          J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          hJ : LE.le J I₀
          j✝ : Fin (HAdd.hAdd n 1)
          x : Real
          j : Fin n
          hx : Membership.mem (Set.Ioo (J.lower (i.succAbove j)) (J.upper (i.succAbove j …
          this : LE.le ↑(J.face i) ↑(I₀.face i)
          ⊢ Eq (HAdd.hAdd (Option.elim' 0 (fun J => HSub.hSub (f (J.upper i) (J.face i)) …
        -/
        rw [le_iff_Icc, @Box.Icc_eq_pi _ I₀] at hJ
        /-
          case refine_2
          ι : Type u_1
          M : Type u_2
          n : Nat
          N : Type u_3
          inst✝² : AddCommMonoid M
          inst✝¹ : AddCommMonoid N
          I₀✝ : WithTop (BoxIntegral.Box ι)
          I : BoxIntegral.Box ι
          i✝ : ι
          G : Type u
          inst✝ : AddCommGroup G
          I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 1)
          f : Real → BoxIntegral.Box (Fin n) → G
          fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
          hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
          J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          hJ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Set.univ.pi fun i => Set.Icc (I …
          j✝ : Fin (HAdd.hAdd n 1)
          x : Real
          j : Fin n
          hx : Membership.mem (Set.Ioo (J.lower (i.succAbove j)) (J.upper (i.succAbove j …
          this : LE.le ↑(J.face i) ↑(I₀.face i)
          ⊢ Eq (HAdd.hAdd (Option.elim' 0 (fun J => HSub.hSub (f (J.upper i) (J.face i)) …
        -/
        simp only
        rw [hf _ (hJ J.upper_mem_Icc _ trivial), hf _ (hJ J.lower_mem_Icc _ trivial),
          ← (fb _).map_split_add this j x, ← (fb _).map_split_add this j x]
        /-
          case refine_2
          ι : Type u_1
          M : Type u_2
          n : Nat
          N : Type u_3
          inst✝² : AddCommMonoid M
          inst✝¹ : AddCommMonoid N
          I₀✝ : WithTop (BoxIntegral.Box ι)
          I : BoxIntegral.Box ι
          i✝ : ι
          G : Type u
          inst✝ : AddCommGroup G
          I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 1)
          f : Real → BoxIntegral.Box (Fin n) → G
          fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
          hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
          J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          hJ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Set.univ.pi fun i => Set.Icc (I …
          j✝ : Fin (HAdd.hAdd n 1)
          x : Real
          j : Fin n
          hx : Membership.mem (Set.Ioo (J.lower (i.succAbove j)) (J.upper (i.succAbove j …
          this : LE.le ↑(J.face i) ↑(I₀.face i)
          ⊢ Eq (HAdd.hAdd (Option.elim' 0 (fun J => HSub.hSub (f (J.upper i) (J.face i)) …
        -/
        have hx' : x ∈ Ioo ((J.face i).lower j) ((J.face i).upper j) := hx
        simp only [Box.splitLower_def hx, Box.splitUpper_def hx, Box.splitLower_def hx',
          Box.splitUpper_def hx', ← WithBot.some_eq_coe, Option.elim', Box.face_mk,
          update_of_ne (Fin.succAbove_ne _ _).symm, sub_add_sub_comm,
          update_comp_eq_of_injective _ (Fin.strictMono_succAbove i).injective j x, ← hf]
        /-
          case refine_2
          ι : Type u_1
          M : Type u_2
          n : Nat
          N : Type u_3
          inst✝² : AddCommMonoid M
          inst✝¹ : AddCommMonoid N
          I₀✝ : WithTop (BoxIntegral.Box ι)
          I : BoxIntegral.Box ι
          i✝ : ι
          G : Type u
          inst✝ : AddCommGroup G
          I₀ : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 1)
          f : Real → BoxIntegral.Box (Fin n) → G
          fb : ↑(Set.Icc (I₀.lower i) (I₀.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) …
          hf : ∀ (x : Real) (hx : Membership.mem (Set.Icc (I₀.lower i) (I₀.upper i)) x)  …
          J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
          hJ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Set.univ.pi fun i => Set.Icc (I …
          j✝ : Fin (HAdd.hAdd n 1)
          x : Real
          j : Fin n
          hx : Membership.mem (Set.Ioo (J.lower (i.succAbove j)) (J.upper (i.succAbove j …
          this : LE.le ↑(J.face i) ↑(I₀.face i)
          hx' : Membership.mem (Set.Ioo ((J.face i).lower j) ((J.face i).upper j)) x
          ⊢ Eq (HSub.hSub (HAdd.hAdd (f (J.upper i) { lower := Function.comp J.lower i.s …
        -/
        simp only [Box.face])
        /-
          🎉 no goals
        -/


