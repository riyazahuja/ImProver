/-- A proper cone is a pointed cone `K` that is closed. Proper cones have the nice property that
they are equal to their double dual, see `ProperCone.dual_dual`.
This makes them useful for defining cone programs and proving duality theorems. -/
structure ProperCone (𝕜 : Type*) (E : Type*) [OrderedSemiring 𝕜] [AddCommMonoid E]
    [TopologicalSpace E] [Module 𝕜 E] extends Submodule {c : 𝕜 // 0 ≤ c} E where
  isClosed' : IsClosed (carrier : Set E)


/-- A `PointedCone` is defined as an alias of submodule. We replicate the abbreviation here and
define `toPointedCone` as an alias of `toSubmodule`. -/
abbrev toPointedCone (C : ProperCone 𝕜 E) := C.toSubmodule


instance : Coe (ProperCone 𝕜 E) (PointedCone 𝕜 E) :=
  ⟨toPointedCone⟩

-- Porting note: now a syntactic tautology
-- @[simp]
-- theorem toConvexCone_eq_coe (K : ProperCone 𝕜 E) : K.toConvexCone = K :=
--   rfl


theorem toPointedCone_injective : Function.Injective ((↑) : ProperCone 𝕜 E → PointedCone 𝕜 E) :=
                  /-
                    𝕜 : Type u_1
                    inst✝³ : OrderedSemiring 𝕜
                    E : Type u_2
                    inst✝² : AddCommMonoid E
                    inst✝¹ : TopologicalSpace E
                    inst✝ : Module 𝕜 E
                    S T : ProperCone 𝕜 E
                    h : Eq ↑S ↑T
                    ⊢ Eq S T
                  -/
  fun S T h => by cases S; cases T; congr
                                    /-
                                      🎉 no goals
                                    -/

-- TODO: add `ConvexConeClass` that extends `SetLike` and replace the below instance

instance : SetLike (ProperCone 𝕜 E) E where
  coe K := K.carrier
  coe_injective' _ _ h := ProperCone.toPointedCone_injective (SetLike.coe_injective h)


@[ext]
theorem ext {S T : ProperCone 𝕜 E} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


@[simp]
theorem mem_coe {x : E} {K : ProperCone 𝕜 E} : x ∈ (K : PointedCone 𝕜 E) ↔ x ∈ K :=
  Iff.rfl


instance instZero (K : ProperCone 𝕜 E) : Zero K := PointedCone.instZero (K.toSubmodule)


protected theorem nonempty (K : ProperCone 𝕜 E) : (K : Set E).Nonempty :=
         /-
           𝕜 : Type u_1
           inst✝³ : OrderedSemiring 𝕜
           E : Type u_2
           inst✝² : AddCommMonoid E
           inst✝¹ : TopologicalSpace E
           inst✝ : Module 𝕜 E
           K : ProperCone 𝕜 E
           ⊢ Membership.mem (↑K) 0
         -/
  ⟨0, by { simp_rw [SetLike.mem_coe, ← ProperCone.mem_coe, Submodule.zero_mem] }⟩
         /-
           🎉 no goals
         -/


protected theorem isClosed (K : ProperCone 𝕜 E) : IsClosed (K : Set E) :=
  K.isClosed'


/-- The positive cone is the proper cone formed by the set of nonnegative elements in an ordered
module. -/
def positive : ProperCone 𝕜 E where
  toSubmodule := PointedCone.positive 𝕜 E
  isClosed' := isClosed_Ici


@[simp]
theorem mem_positive {x : E} : x ∈ positive 𝕜 E ↔ 0 ≤ x :=
  Iff.rfl


@[simp]
theorem coe_positive : ↑(positive 𝕜 E) = ConvexCone.positive 𝕜 E :=
  rfl


instance : Zero (ProperCone 𝕜 E) :=
  ⟨{ toSubmodule := 0
     isClosed' := isClosed_singleton }⟩


instance : Inhabited (ProperCone 𝕜 E) :=
  ⟨0⟩


@[simp]
theorem mem_zero (x : E) : x ∈ (0 : ProperCone 𝕜 E) ↔ x = 0 :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_zero : ↑(0 : ProperCone 𝕜 E) = (0 : ConvexCone 𝕜 E) :=
  rfl


theorem pointed_zero : ((0 : ProperCone 𝕜 E) : ConvexCone 𝕜 E).Pointed := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : OrderedSemiring 𝕜
    E : Type u_2
    inst✝³ : AddCommMonoid E
    inst✝² : TopologicalSpace E
    inst✝¹ : T1Space E
    inst✝ : Module 𝕜 E
    ⊢ (↑↑0).Pointed
  -/
  simp [ConvexCone.pointed_zero]
  /-
    🎉 no goals
  -/


protected theorem pointed (K : ProperCone ℝ E) : (K : ConvexCone ℝ E).Pointed :=
  (K : ConvexCone ℝ E).pointed_of_nonempty_of_isClosed K.nonempty K.isClosed


/-- The closure of image of a proper cone under a continuous `ℝ`-linear map is a proper cone. We
use continuous maps here so that the comap of f is also a map between proper cones. -/
noncomputable def map (f : E →L[ℝ] F) (K : ProperCone ℝ E) : ProperCone ℝ F where
  toSubmodule := PointedCone.closure (PointedCone.map (f : E →ₗ[ℝ] F) ↑K)
  isClosed' := isClosed_closure


@[simp, norm_cast]
theorem coe_map (f : E →L[ℝ] F) (K : ProperCone ℝ E) :
    ↑(K.map f) = (PointedCone.map (f : E →ₗ[ℝ] F) ↑K).closure :=
  rfl


@[simp]
theorem mem_map {f : E →L[ℝ] F} {K : ProperCone ℝ E} {y : F} :
    y ∈ K.map f ↔ y ∈ (PointedCone.map (f : E →ₗ[ℝ] F) ↑K).closure :=
  Iff.rfl


@[simp]
theorem map_id (K : ProperCone ℝ E) : K.map (ContinuousLinearMap.id ℝ E) = K :=
                                           /-
                                             E : Type u_1
                                             inst✝¹ : NormedAddCommGroup E
                                             inst✝ : InnerProductSpace Real E
                                             K : ProperCone Real E
                                             ⊢ Eq ↑(ProperCone.map (ContinuousLinearMap.id Real E) K) ↑K
                                           -/
  ProperCone.toPointedCone_injective <| by simpa using IsClosed.closure_eq K.isClosed
                                           /-
                                             🎉 no goals
                                           -/


/-- The inner dual cone of a proper cone is a proper cone. -/
def dual (K : ProperCone ℝ E) : ProperCone ℝ E where
  toSubmodule := PointedCone.dual (K : PointedCone ℝ E)
  isClosed' := isClosed_innerDualCone _


@[simp, norm_cast]
theorem coe_dual (K : ProperCone ℝ E) : K.dual = (K : Set E).innerDualCone :=
  rfl


open scoped InnerProductSpace in
@[simp]
theorem mem_dual {K : ProperCone ℝ E} {y : E} : y ∈ dual K ↔ ∀ ⦃x⦄, x ∈ K → 0 ≤ ⟪x, y⟫_ℝ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    K : ProperCone Real E
    y : E
    ⊢ Iff (Membership.mem K.dual y) (∀ ⦃x : E⦄, Membership.mem K x → LE.le 0 (Inne …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- The preimage of a proper cone under a continuous `ℝ`-linear map is a proper cone. -/
noncomputable def comap (f : E →L[ℝ] F) (S : ProperCone ℝ F) : ProperCone ℝ E where
  toSubmodule := PointedCone.comap (f : E →ₗ[ℝ] F) S
  isClosed' := by
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace Real E
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : InnerProductSpace Real F
      G : Type u_3
      inst✝¹ : NormedAddCommGroup G
      inst✝ : InnerProductSpace Real G
      f : ContinuousLinearMap (RingHom.id Real) E F
      S : ProperCone Real F
      ⊢ IsClosed (PointedCone.comap ↑f ↑S).carrier
    -/
    rw [PointedCone.comap]
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace Real E
      F : Type u_2
      inst✝³ : NormedAddCommGroup F
      inst✝² : InnerProductSpace Real F
      G : Type u_3
      inst✝¹ : NormedAddCommGroup G
      inst✝ : InnerProductSpace Real G
      f : ContinuousLinearMap (RingHom.id Real) E F
      S : ProperCone Real F
      ⊢ IsClosed (Submodule.comap (↑(Subtype fun c => LE.le 0 c) ↑f) ↑S).carrier
    -/
    apply IsClosed.preimage f.2 S.isClosed
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_comap (f : E →L[ℝ] F) (S : ProperCone ℝ F) : (S.comap f : Set E) = f ⁻¹' S :=
  rfl


@[simp]
theorem comap_id (S : ConvexCone ℝ E) : S.comap LinearMap.id = S :=
  SetLike.coe_injective preimage_id


theorem comap_comap (g : F →L[ℝ] G) (f : E →L[ℝ] F) (S : ProperCone ℝ G) :
    (S.comap g).comap f = S.comap (g.comp f) :=
                              /-
                                E : Type u_1
                                inst✝⁵ : NormedAddCommGroup E
                                inst✝⁴ : InnerProductSpace Real E
                                F : Type u_2
                                inst✝³ : NormedAddCommGroup F
                                inst✝² : InnerProductSpace Real F
                                G : Type u_3
                                inst✝¹ : NormedAddCommGroup G
                                inst✝ : InnerProductSpace Real G
                                g : ContinuousLinearMap (RingHom.id Real) F G
                                f : ContinuousLinearMap (RingHom.id Real) E F
                                S : ProperCone Real G
                                ⊢ Eq ↑(ProperCone.comap f (ProperCone.comap g S)) ↑(ProperCone.comap (g.comp f …
                              -/
  SetLike.coe_injective <| by congr
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem mem_comap {f : E →L[ℝ] F} {S : ProperCone ℝ F} {x : E} : x ∈ S.comap f ↔ f x ∈ S :=
  Iff.rfl


/-- The dual of the dual of a proper cone is itself. -/
@[simp]
theorem dual_dual (K : ProperCone ℝ E) : K.dual.dual = K :=
  ProperCone.toPointedCone_injective <| PointedCone.toConvexCone_injective <|
    (K : ConvexCone ℝ E).innerDualCone_of_innerDualCone_eq_self K.nonempty K.isClosed


/-- This is a relative version of
`ConvexCone.hyperplane_separation_of_nonempty_of_isClosed_of_nmem`, which we recover by setting
`f` to be the identity map. This is also a geometric interpretation of the Farkas' lemma
stated using proper cones. -/
theorem hyperplane_separation (K : ProperCone ℝ E) {f : E →L[ℝ] F} {b : F} :
    b ∈ K.map f ↔ ∀ y : F, adjoint f y ∈ K.dual → 0 ≤ ⟪y, b⟫_ℝ :=
  Iff.intro
    (by
      -- suppose `b ∈ K.map f`
      simp_rw [mem_map, PointedCone.mem_closure, PointedCone.coe_map, coe_coe,
        mem_closure_iff_seq_limit, mem_image, SetLike.mem_coe, mem_coe, mem_dual,
        adjoint_inner_right, forall_exists_index, and_imp]

      -- there is a sequence `seq : ℕ → F` in the image of `f` that converges to `b`
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        ⊢ ∀ (x : Nat → F), (∀ (n : Nat), Exists fun x_1 => And (Membership.mem K x_1)  …
      -/
      rintro seq hmem htends y hinner
      suffices h : ∀ n, 0 ≤ ⟪y, seq n⟫_ℝ from
        ge_of_tendsto'
          (Continuous.seqContinuous (Continuous.inner (@continuous_const _ _ _ _ y) continuous_id)
            htends)
          h
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        seq : Nat → F
        hmem : ∀ (n : Nat), Exists fun x => And (Membership.mem K x) (Eq (f x) (seq n))
        htends : Filter.Tendsto seq Filter.atTop (nhds b)
        y : F
        hinner : ∀ ⦃x : E⦄, Membership.mem K x → LE.le 0 (Inner.inner (f x) y)
        ⊢ ∀ (n : Nat), LE.le 0 (Inner.inner y (seq n))
      -/
      intro n
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        seq : Nat → F
        hmem : ∀ (n : Nat), Exists fun x => And (Membership.mem K x) (Eq (f x) (seq n))
        htends : Filter.Tendsto seq Filter.atTop (nhds b)
        y : F
        hinner : ∀ ⦃x : E⦄, Membership.mem K x → LE.le 0 (Inner.inner (f x) y)
        n : Nat
        ⊢ LE.le 0 (Inner.inner y (seq n))
      -/
      obtain ⟨_, h, hseq⟩ := hmem n
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        seq : Nat → F
        hmem : ∀ (n : Nat), Exists fun x => And (Membership.mem K x) (Eq (f x) (seq n))
        htends : Filter.Tendsto seq Filter.atTop (nhds b)
        y : F
        hinner : ∀ ⦃x : E⦄, Membership.mem K x → LE.le 0 (Inner.inner (f x) y)
        n : Nat
        w✝ : E
        h : Membership.mem K w✝
        hseq : Eq (f w✝) (seq n)
        ⊢ LE.le 0 (Inner.inner y (seq n))
      -/
      simpa only [← hseq, real_inner_comm] using hinner h)
      /-
        🎉 no goals
      -/
    (by
      -- proof by contradiction
      -- suppose `b ∉ K.map f`
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        ⊢ (∀ (y : F), Membership.mem K.dual ((ContinuousLinearMap.adjoint f) y) → LE.l …
      -/
      intro h
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : ∀ (y : F), Membership.mem K.dual ((ContinuousLinearMap.adjoint f) y) → LE. …
        ⊢ Membership.mem (ProperCone.map f K) b
      -/
      contrapose! h

      -- as `b ∉ K.map f`, there is a hyperplane `y` separating `b` from `K.map f`
      /-
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        ⊢ Exists fun y => And (Membership.mem K.dual ((ContinuousLinearMap.adjoint f)  …
      -/
      let C := @PointedCone.toConvexCone ℝ F _ _ _ (K.map f)
      obtain ⟨y, hxy, hyb⟩ :=
        @ConvexCone.hyperplane_separation_of_nonempty_of_isClosed_of_nmem
        _ _ _ _ C (K.map f).nonempty (K.map f).isClosed b h

      -- the rest of the proof is a straightforward algebraic manipulation
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        ⊢ Exists fun y => And (Membership.mem K.dual ((ContinuousLinearMap.adjoint f)  …
      -/
      refine ⟨y, ?_, hyb⟩
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        ⊢ Membership.mem K.dual ((ContinuousLinearMap.adjoint f) y)
      -/
      simp_rw [ProperCone.mem_dual, adjoint_inner_right]
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        ⊢ ∀ ⦃x : E⦄, Membership.mem K x → LE.le 0 (Inner.inner (f x) y)
      -/
      intro x hxK
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        x : E
        hxK : Membership.mem K x
        ⊢ LE.le 0 (Inner.inner (f x) y)
      -/
      apply hxy (f x)
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        x : E
        hxK : Membership.mem K x
        ⊢ Membership.mem C (f x)
      -/
      simp_rw [C, coe_map]
      /-
        case intro.intro
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        x : E
        hxK : Membership.mem K x
        ⊢ Membership.mem (↑(PointedCone.map ↑f ↑K).closure) (f x)
      -/
      apply subset_closure
      simp_rw [PointedCone.toConvexCone_map, ConvexCone.coe_map, coe_coe, mem_image,
        SetLike.mem_coe]
      /-
        case intro.intro.a
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace Real E
        inst✝³ : CompleteSpace E
        F : Type u_2
        inst✝² : NormedAddCommGroup F
        inst✝¹ : InnerProductSpace Real F
        inst✝ : CompleteSpace F
        K : ProperCone Real E
        f : ContinuousLinearMap (RingHom.id Real) E F
        b : F
        h : Not (Membership.mem (ProperCone.map f K) b)
        C : ConvexCone Real F := ↑↑(ProperCone.map f K)
        y : F
        hxy : ∀ (x : F), Membership.mem C x → LE.le 0 (Inner.inner x y)
        hyb : LT.lt (Inner.inner y b) 0
        x : E
        hxK : Membership.mem K x
        ⊢ Exists fun x_1 => And (Membership.mem (↑↑K) x_1) (Eq (f x_1) (f x))
      -/
      exact ⟨x, hxK, rfl⟩)
      /-
        🎉 no goals
      -/


theorem hyperplane_separation_of_nmem (K : ProperCone ℝ E) {f : E →L[ℝ] F} {b : F}
    (disj : b ∉ K.map f) : ∃ y : F, adjoint f y ∈ K.dual ∧ ⟪y, b⟫_ℝ < 0 := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace Real E
    inst✝³ : CompleteSpace E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : CompleteSpace F
    K : ProperCone Real E
    f : ContinuousLinearMap (RingHom.id Real) E F
    b : F
    disj : Not (Membership.mem (ProperCone.map f K) b)
    ⊢ Exists fun y => And (Membership.mem K.dual ((ContinuousLinearMap.adjoint f)  …
  -/
  contrapose! disj; rwa [K.hyperplane_separation]
                    /-
                      🎉 no goals
                    -/


