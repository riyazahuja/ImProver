/-- The valuation on a nonarchimedean normed field `K` defined as `nnnorm`. -/
def valuation : Valuation K ℝ≥0 where
  toFun           := nnnorm
  map_zero'       := nnnorm_zero
  map_one'        := nnnorm_one
  map_mul'        := nnnorm_mul
  map_add_le_max' := IsUltrametricDist.norm_add_le_max


@[simp]
theorem valuation_apply (x : K) : valuation x = ‖x‖₊ := rfl


/-- The valued field structure on a nonarchimedean normed field `K`, determined by the norm. -/
def toValued : Valued K ℝ≥0 :=
  { hK.toUniformSpace,
    @NonUnitalNormedRing.toNormedAddCommGroup K _ with
    v := valuation
    is_topological_valuation := fun U => by
      /-
        K : Type u_1
        hK : NormedField K
        inst✝ : IsUltrametricDist K
        U : Set K
        ⊢ Iff (Membership.mem (nhds 0) U) (Exists fun γ => HasSubset.Subset (setOf fun …
      -/
      rw [Metric.mem_nhds_iff]
      exact ⟨fun ⟨ε, hε, h⟩  =>
          ⟨Units.mk0 ⟨ε, le_of_lt hε⟩ (ne_of_gt hε), fun x hx ↦ h (mem_ball_zero_iff.mpr hx)⟩,
        fun ⟨ε, hε⟩ => ⟨(ε : ℝ), NNReal.coe_pos.mpr (Units.zero_lt _),
          fun x hx ↦ hε (mem_ball_zero_iff.mp hx)⟩⟩ }


instance {K : Type*} [NontriviallyNormedField K] [IsUltrametricDist K] :
    Valuation.RankOne (valuation (K := K)) where
  hom := .id _
  strictMono' := strictMono_id
  nontrivial' := (exists_one_lt_norm K).imp fun x h ↦ by
    /-
      K✝ : Type u_1
      hK : NormedField K✝
      inst✝² : IsUltrametricDist K✝
      K : Type u_2
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      x : K
      h : LT.lt 1 (Norm.norm x)
      ⊢ And (Ne (NormedField.valuation x) 0) (Ne (NormedField.valuation x) 1)
    -/
    have h' : x ≠ 0 := norm_eq_zero.not.mp (h.gt.trans' (by simp)).ne'
    /-
      K✝ : Type u_1
      hK : NormedField K✝
      inst✝² : IsUltrametricDist K✝
      K : Type u_2
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      x : K
      h : LT.lt 1 (Norm.norm x)
      h' : Ne x 0
      ⊢ And (Ne (NormedField.valuation x) 0) (Ne (NormedField.valuation x) 1)
    -/
    simp [valuation_apply, ← NNReal.coe_inj, h.ne', h']
    /-
      🎉 no goals
    -/


/-- The norm function determined by a rank one valuation on a field `L`. -/
def norm : L → ℝ := fun x : L => hv.hom (Valued.v x)


                                               /-
                                                 L : Type u_1
                                                 inst✝¹ : Field L
                                                 Γ₀ : Type u_2
                                                 inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                                 val : Valued L Γ₀
                                                 hv : Valued.v.RankOne
                                                 x : L
                                                 ⊢ LE.le 0 (Valued.norm x)
                                               -/
theorem norm_nonneg (x : L) : 0 ≤ norm x := by simp only [norm, NNReal.zero_le_coe]
                                               /-
                                                 🎉 no goals
                                               -/


theorem norm_add_le (x y : L) : norm (x + y) ≤ max (norm x) (norm y) := by
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x y : L
    ⊢ LE.le (Valued.norm (HAdd.hAdd x y)) (Max.max (Valued.norm x) (Valued.norm y))
  -/
  simp only [norm, NNReal.coe_le_coe, le_max_iff, StrictMono.le_iff_le hv.strictMono]
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x y : L
    ⊢ Or (LE.le (Valued.v (HAdd.hAdd x y)) (Valued.v x)) (LE.le (Valued.v (HAdd.hA …
  -/
  exact le_max_iff.mp (Valuation.map_add_le_max' val.v _ _)
  /-
    🎉 no goals
  -/


theorem norm_eq_zero {x : L} (hx : norm x = 0) : x = 0 := by
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x : L
    hx : Eq (Valued.norm x) 0
    ⊢ Eq x 0
  -/
  simpa [norm, NNReal.coe_eq_zero, RankOne.hom_eq_zero_iff, zero_iff] using hx
  /-
    🎉 no goals
  -/


/-- The normed field structure determined by a rank one valuation. -/
def toNormedField : NormedField L :=
  { (inferInstance : Field L) with
    norm := norm
    dist := fun x y => norm (x - y)
    dist_self := fun x => by
      /-
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        x : L
        ⊢ Eq (Dist.dist x x) 0
      -/
      simp only [sub_self, norm, Valuation.map_zero, hv.hom.map_zero, NNReal.coe_zero]
      /-
        🎉 no goals
      -/
                               /-
                                 L : Type u_1
                                 inst✝¹ : Field L
                                 Γ₀ : Type u_2
                                 inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                 val : Valued L Γ₀
                                 hv : Valued.v.RankOne
                                 x y : L
                                 ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                               -/
    dist_comm := fun x y => by simp only [norm]; rw [← neg_sub, Valuation.map_neg]
                                                 /-
                                                   🎉 no goals
                                                 -/
    dist_triangle := fun x y z => by
      /-
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        x y z : L
        ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
      -/
      simp only [← sub_add_sub_cancel x y z]
      exact le_trans (norm_add_le _ _)
        (max_le_add_of_nonneg (norm_nonneg _) (norm_nonneg _))
    eq_of_dist_eq_zero := fun hxy => eq_of_sub_eq_zero (norm_eq_zero hxy)
    dist_eq := fun x y => rfl
                               /-
                                 L : Type u_1
                                 inst✝¹ : Field L
                                 Γ₀ : Type u_2
                                 inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                 val : Valued L Γ₀
                                 hv : Valued.v.RankOne
                                 x y : L
                                 ⊢ Eq (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
                               -/
    norm_mul' := fun x y => by simp only [norm, ← NNReal.coe_mul, _root_.map_mul]
                               /-
                                 🎉 no goals
                               -/
      /-
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        ⊢ Eq (uniformity L) (iInf fun ε => iInf fun h => Filter.principal (setOf fun p …
      -/
    toUniformSpace := Valued.toUniformSpace
      /-
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        this : Nonempty (Subtype fun ε => GT.gt ε 0)
        ⊢ Eq (uniformity L) (iInf fun ε => iInf fun h => Filter.principal (setOf fun p …
      -/
    uniformity_dist := by
      /-
        case h
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        this : Nonempty (Subtype fun ε => GT.gt ε 0)
        U : Set (Prod L L)
        ⊢ Iff (Membership.mem (uniformity L) U) (Membership.mem (iInf fun ε => iInf fu …
      -/
      haveI : Nonempty { ε : ℝ // ε > 0 } := nonempty_Ioi_subtype
        /-
          case h
          L : Type u_1
          inst✝¹ : Field L
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          val : Valued L Γ₀
          hv : Valued.v.RankOne
          this : Nonempty (Subtype fun ε => GT.gt ε 0)
          U : Set (Prod L L)
          ⊢ Iff (Exists fun i => And True (HasSubset.Subset (setOf fun p => LT.lt (Value …
        -/
      ext U
        /-
          case h
          L : Type u_1
          inst✝¹ : Field L
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          val : Valued L Γ₀
          hv : Valued.v.RankOne
          this : Nonempty (Subtype fun ε => GT.gt ε 0)
          U : Set (Prod L L)
          ⊢ Iff (Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub. …
        -/
      rw [hasBasis_iff.mp (Valued.hasBasis_uniformity L Γ₀), iInf_subtype', mem_iInf_of_directed]
          /-
            case h.refine_1
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            ⊢ Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt (Val …
          -/
      · simp only [true_and, mem_principal, Subtype.exists, gt_iff_lt, exists_prop]
        refine ⟨fun ⟨ε, hε⟩ => ?_, fun ⟨r, hr_pos, hr⟩ => ?_⟩
        · set δ : ℝ≥0 := hv.hom ε with hδ
          have hδ_pos : 0 < δ := by
          /-
            case h.refine_1
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            δ : NNReal := (Valuation.RankOne.hom Valued.v) ↑ε
            hδ : Eq δ ((Valuation.RankOne.hom Valued.v) ↑ε)
            hδ_pos : LT.lt 0 δ
            ⊢ Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt (Val …
          -/
            rw [hδ, ← _root_.map_zero hv.hom]
          /-
            case right
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            δ : NNReal := (Valuation.RankOne.hom Valued.v) ↑ε
            hδ : Eq δ ((Valuation.RankOne.hom Valued.v) ↑ε)
            hδ_pos : LT.lt 0 δ
            ⊢ HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2)) ↑δ) U
          -/
            exact hv.strictMono _ (Units.zero_lt ε)
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            δ : NNReal := (Valuation.RankOne.hom Valued.v) ↑ε
            hδ : Eq δ ((Valuation.RankOne.hom Valued.v) ↑ε)
            hδ_pos : LT.lt 0 δ
            ⊢ HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2)) ↑δ) …
          -/
          use δ, hδ_pos
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            δ : NNReal := (Valuation.RankOne.hom Valued.v) ↑ε
            hδ : Eq δ ((Valuation.RankOne.hom Valued.v) ↑ε)
            hδ_pos : LT.lt 0 δ
            x : Prod L L
            hx : Membership.mem (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2)) ↑δ …
            ⊢ Membership.mem (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) x
          -/
          apply subset_trans _ hε
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            δ : NNReal := (Valuation.RankOne.hom Valued.v) ↑ε
            hδ : Eq δ ((Valuation.RankOne.hom Valued.v) ↑ε)
            hδ_pos : LT.lt 0 δ
            x : Prod L L
            hx : LT.lt ((Valuation.RankOne.hom Valued.v) (Valued.v (HSub.hSub x.1 x.2))) ( …
            ⊢ Membership.mem (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) x
          -/
          intro x hx
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hS …
            ε : Units Γ₀
            hε : HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑ε) U
            δ : NNReal := (Valuation.RankOne.hom Valued.v) ↑ε
            hδ : Eq δ ((Valuation.RankOne.hom Valued.v) ↑ε)
            hδ_pos : LT.lt 0 δ
            x : Prod L L
            hx : LT.lt ((Valuation.RankOne.hom Valued.v) (Valued.v (HSub.hSub x.1 x.2))) ( …
            ⊢ LT.lt (Valued.v (HSub.hSub x.1 x.2)) ↑ε
          -/
          simp only [mem_setOf_eq, norm, hδ, NNReal.val_eq_coe, NNReal.coe_lt_coe] at hx
          /-
            🎉 no goals
          -/
          rw [mem_setOf, ← neg_sub, Valuation.map_neg]
          exact (RankOne.strictMono Valued.v).lt_iff_lt.mp hx
          /-
            case h.refine_2
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            ⊢ Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub  …
          -/
        · haveI : Nontrivial Γ₀ˣ := (nontrivial_iff_exists_ne (1 : Γ₀ˣ)).mpr
          /-
            case h.refine_2.intro
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            ⊢ Exists fun i => HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub  …
          -/
            ⟨RankOne.unit val.v, RankOne.unit_ne_one val.v⟩
          /-
            case h
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            ⊢ HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑u) U
          -/
          obtain ⟨u, hu⟩ := Real.exists_lt_of_strictMono hv.strictMono hr_pos
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            ⊢ HasSubset.Subset (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑u) (s …
          -/
          use u
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            x : Prod L L
            hx : Membership.mem (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑u) x
            ⊢ Membership.mem (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2)) r) x
          -/
          apply subset_trans _ hr
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            x : Prod L L
            hx : Membership.mem (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑u) x
            ⊢ LT.lt (↑((Valuation.RankOne.hom Valued.v) (Valued.v (HSub.hSub x.1 x.2)))) r
          -/
          intro x hx
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            x : Prod L L
            hx : Membership.mem (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑u) x
            ⊢ LT.lt ↑((Valuation.RankOne.hom Valued.v) (Valued.v (HSub.hSub x.1 x.2))) ↑(( …
          -/
          simp only [norm, mem_setOf_eq]
          /-
            L : Type u_1
            inst✝¹ : Field L
            Γ₀ : Type u_2
            inst✝ : LinearOrderedCommGroupWithZero Γ₀
            val : Valued L Γ₀
            hv : Valued.v.RankOne
            this✝ : Nonempty (Subtype fun ε => GT.gt ε 0)
            U : Set (Prod L L)
            x✝ : Exists fun a => And (LT.lt 0 a) (HasSubset.Subset (setOf fun p => LT.lt ( …
            r : Real
            hr_pos : LT.lt 0 r
            hr : HasSubset.Subset (setOf fun p => LT.lt (Valued.norm (HSub.hSub p.1 p.2))  …
            this : Nontrivial (Units Γ₀)
            u : Units Γ₀
            hu : LT.lt (↑((Valuation.RankOne.hom Valued.v) ↑u)) r
            x : Prod L L
            hx : Membership.mem (setOf fun p => LT.lt (Valued.v (HSub.hSub p.2 p.1)) ↑u) x
            ⊢ LT.lt ((Valuation.RankOne.hom Valued.v) (Valued.v (HSub.hSub x.2 x.1))) ((Va …
          -/
          apply lt_trans _ hu
          /-
            🎉 no goals
          -/
        /-
          case h.h
          L : Type u_1
          inst✝¹ : Field L
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          val : Valued L Γ₀
          hv : Valued.v.RankOne
          this : Nonempty (Subtype fun ε => GT.gt ε 0)
          U : Set (Prod L L)
          ⊢ Directed (fun x1 x2 => GE.ge x1 x2) fun x => Filter.principal (setOf fun p = …
        -/
          rw [NNReal.coe_lt_coe, ← neg_sub, Valuation.map_neg]
        /-
          case h.h
          L : Type u_1
          inst✝¹ : Field L
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          val : Valued L Γ₀
          hv : Valued.v.RankOne
          this : Nonempty (Subtype fun ε => GT.gt ε 0)
          U : Set (Prod L L)
          ⊢ ∀ (x y : Subtype fun ε => GT.gt ε 0), Exists fun z => And (GE.ge (Filter.pri …
        -/
          exact (RankOne.strictMono Valued.v).lt_iff_lt.mpr hx
        /-
          case h.h
          L : Type u_1
          inst✝¹ : Field L
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          val : Valued L Γ₀
          hv : Valued.v.RankOne
          this : Nonempty (Subtype fun ε => GT.gt ε 0)
          U : Set (Prod L L)
          x y : Subtype fun ε => GT.gt ε 0
          ⊢ Exists fun z => And (GE.ge (Filter.principal (setOf fun p => LT.lt (Valued.n …
        -/
      · simp only [Directed]
        /-
          case h
          L : Type u_1
          inst✝¹ : Field L
          Γ₀ : Type u_2
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          val : Valued L Γ₀
          hv : Valued.v.RankOne
          this : Nonempty (Subtype fun ε => GT.gt ε 0)
          U : Set (Prod L L)
          x y : Subtype fun ε => GT.gt ε 0
          ⊢ And (GE.ge (Filter.principal (setOf fun p => LT.lt (Valued.norm (HSub.hSub p …
        -/
        intro x y
        use min x y
        simp only [le_principal_iff, mem_principal, setOf_subset_setOf, Prod.forall]
        exact ⟨fun a b hab => lt_of_lt_of_le hab (min_le_left _ _), fun a b hab =>
            lt_of_lt_of_le hab (min_le_right _ _)⟩ }

-- When a field is valued, one inherits a `NormedField`.
-- Scoped instance to avoid a typeclass loop or non-defeq topology or norms.

protected lemma isNonarchimedean_norm : IsNonarchimedean ((‖·‖): L → ℝ) := Valued.norm_add_le


instance : IsUltrametricDist L :=
  ⟨fun x y z ↦ by
    /-
      L : Type u_1
      inst✝¹ : Field L
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      val : Valued L Γ₀
      hv : Valued.v.RankOne
      x y z : L
      ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
    -/
    refine (Valued.norm_add_le (x - y) (y - z)).trans_eq' ?_
    /-
      L : Type u_1
      inst✝¹ : Field L
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      val : Valued L Γ₀
      hv : Valued.v.RankOne
      x y z : L
      ⊢ Eq (Dist.dist x z) (Valued.norm (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z)))
    -/
    simp only [sub_add_sub_cancel]
    /-
      L : Type u_1
      inst✝¹ : Field L
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      val : Valued L Γ₀
      hv : Valued.v.RankOne
      x y z : L
      ⊢ Eq (Dist.dist x z) (Valued.norm (HSub.hSub x z))
    -/
    rfl ⟩
    /-
      🎉 no goals
    -/


lemma coe_valuation_eq_rankOne_hom_comp_valuation : ⇑NormedField.valuation = hv.hom ∘ val.v := rfl


@[simp]
theorem norm_le_iff : ‖x‖ ≤ ‖x'‖ ↔ val.v x ≤ val.v x' :=
  (Valuation.RankOne.strictMono val.v).le_iff_le


@[simp]
theorem norm_lt_iff : ‖x‖ < ‖x'‖ ↔ val.v x < val.v x' :=
  (Valuation.RankOne.strictMono val.v).lt_iff_lt


@[simp]
theorem norm_le_one_iff : ‖x‖ ≤ 1 ↔ val.v x ≤ 1 := by
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x : L
    ⊢ Iff (LE.le (Norm.norm x) 1) (LE.le (Valued.v x) 1)
  -/
  simpa only [_root_.map_one] using (Valuation.RankOne.strictMono val.v).le_iff_le (b := 1)
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_lt_one_iff : ‖x‖ < 1 ↔ val.v x < 1 := by
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x : L
    ⊢ Iff (LT.lt (Norm.norm x) 1) (LT.lt (Valued.v x) 1)
  -/
  simpa only [_root_.map_one] using (Valuation.RankOne.strictMono val.v).lt_iff_lt (b := 1)
  /-
    🎉 no goals
  -/


@[simp]
theorem one_le_norm_iff : 1 ≤ ‖x‖ ↔ 1 ≤ val.v x := by
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x : L
    ⊢ Iff (LE.le 1 (Norm.norm x)) (LE.le 1 (Valued.v x))
  -/
  simpa only [_root_.map_one] using (Valuation.RankOne.strictMono val.v).le_iff_le (a := 1)
  /-
    🎉 no goals
  -/


@[simp]
theorem one_lt_norm_iff : 1 < ‖x‖ ↔ 1 < val.v x := by
  /-
    L : Type u_1
    inst✝¹ : Field L
    Γ₀ : Type u_2
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    val : Valued L Γ₀
    hv : Valued.v.RankOne
    x : L
    ⊢ Iff (LT.lt 1 (Norm.norm x)) (LT.lt 1 (Valued.v x))
  -/
  simpa only [_root_.map_one] using (Valuation.RankOne.strictMono val.v).lt_iff_lt (a := 1)
  /-
    🎉 no goals
  -/


/--
The nontrivially normed field structure determined by a rank one valuation.
-/
def toNontriviallyNormedField: NontriviallyNormedField L := {
  val.toNormedField with
  non_trivial := by
    /-
      L : Type u_1
      inst✝¹ : Field L
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      val : Valued L Γ₀
      hv : Valued.v.RankOne
      ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
    -/
    obtain ⟨x, hx⟩ := Valuation.RankOne.nontrivial val.v
    /-
      case intro
      L : Type u_1
      inst✝¹ : Field L
      Γ₀ : Type u_2
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      val : Valued L Γ₀
      hv : Valued.v.RankOne
      x : L
      hx : And (Ne (Valued.v x) 0) (Ne (Valued.v x) 1)
      ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
    -/
    rcases Valuation.val_le_one_or_val_inv_le_one val.v x with h | h
      /-
        case intro.inl
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        x : L
        hx : And (Ne (Valued.v x) 0) (Ne (Valued.v x) 1)
        h : LE.le (Valued.v x) 1
        ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
      -/
    · use x⁻¹
      simp only [toNormedField.one_lt_norm_iff, map_inv₀, one_lt_inv₀ (zero_lt_iff.mpr hx.1),
          lt_of_le_of_ne h hx.2]
      /-
        case intro.inr
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        x : L
        hx : And (Ne (Valued.v x) 0) (Ne (Valued.v x) 1)
        h : LE.le (Valued.v (Inv.inv x)) 1
        ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
      -/
    · use x
      /-
        case h
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        x : L
        hx : And (Ne (Valued.v x) 0) (Ne (Valued.v x) 1)
        h : LE.le (Valued.v (Inv.inv x)) 1
        ⊢ LT.lt 1 (Norm.norm x)
      -/
      simp only [map_inv₀, inv_le_one₀ <| zero_lt_iff.mpr hx.1] at h
      /-
        case h
        L : Type u_1
        inst✝¹ : Field L
        Γ₀ : Type u_2
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        val : Valued L Γ₀
        hv : Valued.v.RankOne
        x : L
        hx : And (Ne (Valued.v x) 0) (Ne (Valued.v x) 1)
        h : LE.le 1 (Valued.v x)
        ⊢ LT.lt 1 (Norm.norm x)
      -/
      simp only [toNormedField.one_lt_norm_iff, lt_of_le_of_ne h hx.2.symm]
      /-
        🎉 no goals
      -/
}


