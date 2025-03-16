/-- The definition of the norm on the quotient by an additive subgroup. -/
noncomputable instance normOnQuotient (S : AddSubgroup M) : Norm (M ⧸ S) where
  norm x := sInf (norm '' { m | mk' S m = x })


theorem AddSubgroup.quotient_norm_eq {S : AddSubgroup M} (x : M ⧸ S) :
    ‖x‖ = sInf (norm '' { m : M | (m : M ⧸ S) = x }) :=
  rfl


theorem QuotientAddGroup.norm_eq_infDist {S : AddSubgroup M} (x : M ⧸ S) :
    ‖x‖ = infDist 0 { m : M | (m : M ⧸ S) = x } := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : HasQuotient.Quotient M S
    ⊢ Eq (Norm.norm x) (Metric.infDist 0 (setOf fun m => Eq (↑m) x))
  -/
  simp only [AddSubgroup.quotient_norm_eq, infDist_eq_iInf, sInf_image', dist_zero_left]
  /-
    🎉 no goals
  -/


/-- An alternative definition of the norm on the quotient group: the norm of `((x : M) : M ⧸ S)` is
equal to the distance from `x` to `S`. -/
theorem QuotientAddGroup.norm_mk {S : AddSubgroup M} (x : M) :
    ‖(x : M ⧸ S)‖ = infDist x S := by
  rw [norm_eq_infDist, ← infDist_image (IsometryEquiv.subLeft x).isometry,
    IsometryEquiv.subLeft_apply, sub_zero, ← IsometryEquiv.preimage_symm]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : M
    ⊢ Eq (Metric.infDist x (Set.preimage (⇑(IsometryEquiv.subLeft x).symm) (setOf  …
  -/
  congr 1 with y
  simp only [mem_preimage, IsometryEquiv.subLeft_symm_apply, mem_setOf_eq, QuotientAddGroup.eq,
    neg_add, neg_neg, neg_add_cancel_right, SetLike.mem_coe]


theorem image_norm_nonempty {S : AddSubgroup M} (x : M ⧸ S) :
    (norm '' { m | mk' S m = x }).Nonempty :=
  .image _ <| Quot.exists_rep x


theorem bddBelow_image_norm (s : Set M) : BddBelow (norm '' s) :=
  ⟨0, forall_mem_image.2 fun _ _ ↦ norm_nonneg _⟩


theorem isGLB_quotient_norm {S : AddSubgroup M} (x : M ⧸ S) :
    IsGLB (norm '' { m | mk' S m = x }) (‖x‖) :=
  isGLB_csInf (image_norm_nonempty x) (bddBelow_image_norm _)


/-- The norm on the quotient satisfies `‖-x‖ = ‖x‖`. -/
theorem quotient_norm_neg {S : AddSubgroup M} (x : M ⧸ S) : ‖-x‖ = ‖x‖ := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : HasQuotient.Quotient M S
    ⊢ Eq (Norm.norm (Neg.neg x)) (Norm.norm x)
  -/
  simp only [AddSubgroup.quotient_norm_eq]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : HasQuotient.Quotient M S
    ⊢ Eq (InfSet.sInf (Set.image Norm.norm (setOf fun m => Eq (↑m) (Neg.neg x))))  …
  -/
  congr 1 with r
  /-
    case e_a.h
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : HasQuotient.Quotient M S
    r : Real
    ⊢ Iff (Membership.mem (Set.image Norm.norm (setOf fun m => Eq (↑m) (Neg.neg x) …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> { rintro ⟨m, hm, rfl⟩; use -m; simpa [neg_eq_iff_eq_neg] using hm }
                  /-
                    🎉 no goals
                  -/


theorem quotient_norm_sub_rev {S : AddSubgroup M} (x y : M ⧸ S) : ‖x - y‖ = ‖y - x‖ := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x y : HasQuotient.Quotient M S
    ⊢ Eq (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub y x))
  -/
  rw [← neg_sub, quotient_norm_neg]
  /-
    🎉 no goals
  -/


/-- The norm of the projection is smaller or equal to the norm of the original element. -/
theorem quotient_norm_mk_le (S : AddSubgroup M) (m : M) : ‖mk' S m‖ ≤ ‖m‖ :=
  csInf_le (bddBelow_image_norm _) <| Set.mem_image_of_mem _ rfl


/-- The norm of the projection is smaller or equal to the norm of the original element. -/
theorem quotient_norm_mk_le' (S : AddSubgroup M) (m : M) : ‖(m : M ⧸ S)‖ ≤ ‖m‖ :=
  quotient_norm_mk_le S m


/-- The norm of the image under the natural morphism to the quotient. -/
theorem quotient_norm_mk_eq (S : AddSubgroup M) (m : M) :
    ‖mk' S m‖ = sInf ((‖m + ·‖) '' S) := by
  rw [mk'_apply, norm_mk, sInf_image', ← infDist_image isometry_neg, image_neg_eq_neg,
    neg_coe_set (H := S), infDist_eq_iInf]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    m : M
    ⊢ Eq (iInf fun y => Dist.dist (Neg.neg m) ↑y) (iInf fun a => Norm.norm (HAdd.h …
  -/
  simp only [dist_eq_norm', sub_neg_eq_add, add_comm]
  /-
    🎉 no goals
  -/


/-- The quotient norm is nonnegative. -/
theorem quotient_norm_nonneg (S : AddSubgroup M) (x : M ⧸ S) : 0 ≤ ‖x‖ :=
  Real.sInf_nonneg <| forall_mem_image.2 fun _ _ ↦ norm_nonneg _


/-- The quotient norm is nonnegative. -/
theorem norm_mk_nonneg (S : AddSubgroup M) (m : M) : 0 ≤ ‖mk' S m‖ :=
  quotient_norm_nonneg S _


/-- The norm of the image of `m : M` in the quotient by `S` is zero if and only if `m` belongs
to the closure of `S`. -/
theorem quotient_norm_eq_zero_iff (S : AddSubgroup M) (m : M) :
    ‖mk' S m‖ = 0 ↔ m ∈ closure (S : Set M) := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    m : M
    ⊢ Iff (Eq (Norm.norm ((QuotientAddGroup.mk' S) m)) 0) (Membership.mem (closure …
  -/
  rw [mk'_apply, norm_mk, ← mem_closure_iff_infDist_zero]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    m : M
    ⊢ (↑S).Nonempty
  -/
  exact ⟨0, S.zero_mem⟩
  /-
    🎉 no goals
  -/


theorem QuotientAddGroup.norm_lt_iff {S : AddSubgroup M} {x : M ⧸ S} {r : ℝ} :
    ‖x‖ < r ↔ ∃ m : M, ↑m = x ∧ ‖m‖ < r := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : HasQuotient.Quotient M S
    r : Real
    ⊢ Iff (LT.lt (Norm.norm x) r) (Exists fun m => And (Eq (↑m) x) (LT.lt (Norm.no …
  -/
  rw [isGLB_lt_iff (isGLB_quotient_norm _), exists_mem_image]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x : HasQuotient.Quotient M S
    r : Real
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (setOf fun m => Eq ((QuotientAddG …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- For any `x : M ⧸ S` and any `0 < ε`, there is `m : M` such that `mk' S m = x`
and `‖m‖ < ‖x‖ + ε`. -/
theorem norm_mk_lt {S : AddSubgroup M} (x : M ⧸ S) {ε : ℝ} (hε : 0 < ε) :
    ∃ m : M, mk' S m = x ∧ ‖m‖ < ‖x‖ + ε :=
  norm_lt_iff.1 <| lt_add_of_pos_right _ hε


/-- For any `m : M` and any `0 < ε`, there is `s ∈ S` such that `‖m + s‖ < ‖mk' S m‖ + ε`. -/
theorem norm_mk_lt' (S : AddSubgroup M) (m : M) {ε : ℝ} (hε : 0 < ε) :
    ∃ s ∈ S, ‖m + s‖ < ‖mk' S m‖ + ε := by
  obtain ⟨n : M, hn : mk' S n = mk' S m, hn' : ‖n‖ < ‖mk' S m‖ + ε⟩ :=
    norm_mk_lt (QuotientAddGroup.mk' S m) hε
  /-
    case intro.intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    m : M
    ε : Real
    hε : LT.lt 0 ε
    n : M
    hn : Eq ((QuotientAddGroup.mk' S) n) ((QuotientAddGroup.mk' S) m)
    hn' : LT.lt (Norm.norm n) (HAdd.hAdd (Norm.norm ((QuotientAddGroup.mk' S) m)) ε)
    ⊢ Exists fun s => And (Membership.mem S s) (LT.lt (Norm.norm (HAdd.hAdd m s))  …
  -/
  erw [eq_comm, QuotientAddGroup.eq] at hn
  /-
    case intro.intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    m : M
    ε : Real
    hε : LT.lt 0 ε
    n : M
    hn : Membership.mem S (HAdd.hAdd (Neg.neg m) n)
    hn' : LT.lt (Norm.norm n) (HAdd.hAdd (Norm.norm ((QuotientAddGroup.mk' S) m)) ε)
    ⊢ Exists fun s => And (Membership.mem S s) (LT.lt (Norm.norm (HAdd.hAdd m s))  …
  -/
  use -m + n, hn
  /-
    case right
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    m : M
    ε : Real
    hε : LT.lt 0 ε
    n : M
    hn : Membership.mem S (HAdd.hAdd (Neg.neg m) n)
    hn' : LT.lt (Norm.norm n) (HAdd.hAdd (Norm.norm ((QuotientAddGroup.mk' S) m)) ε)
    ⊢ LT.lt (Norm.norm (HAdd.hAdd m (HAdd.hAdd (Neg.neg m) n))) (HAdd.hAdd (Norm.n …
  -/
  rwa [add_neg_cancel_left]
  /-
    🎉 no goals
  -/


/-- The quotient norm satisfies the triangle inequality. -/
theorem quotient_norm_add_le (S : AddSubgroup M) (x y : M ⧸ S) : ‖x + y‖ ≤ ‖x‖ + ‖y‖ := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x y : HasQuotient.Quotient M S
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  rcases And.intro (mk_surjective x) (mk_surjective y) with ⟨⟨x, rfl⟩, ⟨y, rfl⟩⟩
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x y : M
    ⊢ LE.le (Norm.norm (HAdd.hAdd ↑x ↑y)) (HAdd.hAdd (Norm.norm ↑x) (Norm.norm ↑y))
  -/
  simp only [← mk'_apply, ← map_add, quotient_norm_mk_eq, sInf_image']
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x y : M
    ⊢ LE.le (iInf fun a => Norm.norm (HAdd.hAdd (HAdd.hAdd x y) ↑a)) (HAdd.hAdd (i …
  -/
  refine le_ciInf_add_ciInf fun a b ↦ ?_
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x y : M
    a b : ↑↑S
    ⊢ LE.le (iInf fun a => Norm.norm (HAdd.hAdd (HAdd.hAdd x y) ↑a)) (HAdd.hAdd (N …
  -/
  refine ciInf_le_of_le ⟨0, forall_mem_range.2 fun _ ↦ norm_nonneg _⟩ (a + b) ?_
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    x y : M
    a b : ↑↑S
    ⊢ LE.le (Norm.norm (HAdd.hAdd (HAdd.hAdd x y) ↑(HAdd.hAdd a b))) (HAdd.hAdd (N …
  -/
  exact (congr_arg norm (add_add_add_comm _ _ _ _)).trans_le (norm_add_le _ _)
  /-
    🎉 no goals
  -/


/-- The quotient norm of `0` is `0`. -/
theorem norm_mk_zero (S : AddSubgroup M) : ‖(0 : M ⧸ S)‖ = 0 := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    ⊢ Eq (Norm.norm 0) 0
  -/
  erw [quotient_norm_eq_zero_iff]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    ⊢ Membership.mem (closure ↑S) 0
  -/
  exact subset_closure S.zero_mem
  /-
    🎉 no goals
  -/


/-- If `(m : M)` has norm equal to `0` in `M ⧸ S` for a closed subgroup `S` of `M`, then
`m ∈ S`. -/
theorem norm_mk_eq_zero (S : AddSubgroup M) (hS : IsClosed (S : Set M)) (m : M)
                                      /-
                                        M : Type u_1
                                        inst✝ : SeminormedAddCommGroup M
                                        S : AddSubgroup M
                                        hS : IsClosed ↑S
                                        m : M
                                        h : Eq (Norm.norm ((QuotientAddGroup.mk' S) m)) 0
                                        ⊢ Membership.mem S m
                                      -/
    (h : ‖mk' S m‖ = 0) : m ∈ S := by rwa [quotient_norm_eq_zero_iff, hS.closure_eq] at h
                                      /-
                                        🎉 no goals
                                      -/


theorem quotient_nhd_basis (S : AddSubgroup M) :
    (𝓝 (0 : M ⧸ S)).HasBasis (fun ε ↦ 0 < ε) fun ε ↦ { x | ‖x‖ < ε } := by
  have : ∀ ε : ℝ, mk '' ball (0 : M) ε = { x : M ⧸ S | ‖x‖ < ε } := by
    refine fun ε ↦ Set.ext <| forall_mk.2 fun x ↦ ?_
    rw [ball_zero_eq, mem_setOf_eq, norm_lt_iff, mem_image]
    exact exists_congr fun _ ↦ and_comm
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    this : ∀ (ε : Real), Eq (Set.image QuotientAddGroup.mk (Metric.ball 0 ε)) (set …
    ⊢ (nhds 0).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun x => LT.lt (Norm.n …
  -/
  rw [← QuotientAddGroup.mk_zero, nhds_eq, ← funext this]
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    this : ∀ (ε : Real), Eq (Set.image QuotientAddGroup.mk (Metric.ball 0 ε)) (set …
    ⊢ (Filter.map QuotientAddGroup.mk (nhds 0)).HasBasis (fun ε => LT.lt 0 ε) fun  …
  -/
  exact .map _ Metric.nhds_basis_ball
  /-
    🎉 no goals
  -/


/-- The seminormed group structure on the quotient by an additive subgroup. -/
noncomputable instance AddSubgroup.seminormedAddCommGroupQuotient (S : AddSubgroup M) :
    SeminormedAddCommGroup (M ⧸ S) where
  dist x y := ‖x - y‖
                    /-
                      M : Type u_1
                      N : Type u_2
                      inst✝¹ : SeminormedAddCommGroup M
                      inst✝ : SeminormedAddCommGroup N
                      S : AddSubgroup M
                      x : HasQuotient.Quotient M S
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp only [norm_mk_zero, sub_self]
                    /-
                      🎉 no goals
                    -/
  dist_comm := quotient_norm_sub_rev
  dist_triangle x y z := by
    /-
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      S : AddSubgroup M
      x y z : HasQuotient.Quotient M S
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    refine le_trans ?_ (quotient_norm_add_le _ _ _)
    /-
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      S : AddSubgroup M
      x y z : HasQuotient.Quotient M S
      ⊢ LE.le (Dist.dist x z) (Norm.norm (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z)))
    -/
    exact (congr_arg norm (sub_add_sub_cancel _ _ _).symm).le
    /-
      🎉 no goals
    -/
                       /-
                         M : Type u_1
                         N : Type u_2
                         inst✝¹ : SeminormedAddCommGroup M
                         inst✝ : SeminormedAddCommGroup N
                         S : AddSubgroup M
                         x y : HasQuotient.Quotient M S
                         ⊢ Eq ((fun x y => ↑⟨Norm.norm (HSub.hSub x y), ⋯⟩) x y) (ENNReal.ofReal (Dist. …
                       -/
  edist_dist x y := by exact ENNReal.coe_nnreal_eq _
                       /-
                         🎉 no goals
                       -/
  toUniformSpace := TopologicalAddGroup.toUniformSpace (M ⧸ S)
  uniformity_dist := by
    /-
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      S : AddSubgroup M
      ⊢ Eq (uniformity (HasQuotient.Quotient M S)) (iInf fun ε => iInf fun h => Filt …
    -/
    rw [uniformity_eq_comap_nhds_zero', ((quotient_nhd_basis S).comap _).eq_biInf]
    /-
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      S : AddSubgroup M
      ⊢ Eq (iInf fun i => iInf fun x => Filter.principal (Set.preimage (fun p => HSu …
    -/
    simp only [dist, quotient_norm_sub_rev (Prod.fst _), preimage_setOf_eq]
    /-
      🎉 no goals
    -/

-- This is a sanity check left here on purpose to ensure that potential refactors won't destroy
-- this important property.

/-- The quotient in the category of normed groups. -/
noncomputable instance AddSubgroup.normedAddCommGroupQuotient (S : AddSubgroup M)
    [IsClosed (S : Set M)] : NormedAddCommGroup (M ⧸ S) :=
  { AddSubgroup.seminormedAddCommGroupQuotient S, MetricSpace.ofT0PseudoMetricSpace _ with }

-- This is a sanity check left here on purpose to ensure that potential refactors won't destroy
-- this important property.

/-- The morphism from a seminormed group to the quotient by a subgroup. -/
noncomputable def normedMk (S : AddSubgroup M) : NormedAddGroupHom M (M ⧸ S) :=
  { QuotientAddGroup.mk' S with
                              /-
                                M : Type u_1
                                N : Type u_2
                                inst✝¹ : SeminormedAddCommGroup M
                                inst✝ : SeminormedAddCommGroup N
                                S : AddSubgroup M
                                m : M
                                ⊢ LE.le (Norm.norm ((↑__src✝).toFun m)) (HMul.hMul 1 (Norm.norm m))
                              -/
    bound' := ⟨1, fun m => by simpa [one_mul] using quotient_norm_mk_le _ m⟩ }
                              /-
                                🎉 no goals
                              -/


/-- `S.normedMk` agrees with `QuotientAddGroup.mk' S`. -/
@[simp]
theorem normedMk.apply (S : AddSubgroup M) (m : M) : normedMk S m = QuotientAddGroup.mk' S m :=
  rfl


/-- `S.normedMk` is surjective. -/
theorem surjective_normedMk (S : AddSubgroup M) : Function.Surjective (normedMk S) :=
  Quot.mk_surjective


/-- The kernel of `S.normedMk` is `S`. -/
theorem ker_normedMk (S : AddSubgroup M) : S.normedMk.ker = S :=
  QuotientAddGroup.ker_mk' _


/-- The operator norm of the projection is at most `1`. -/
theorem norm_normedMk_le (S : AddSubgroup M) : ‖S.normedMk‖ ≤ 1 :=
                                                              /-
                                                                M : Type u_1
                                                                inst✝ : SeminormedAddCommGroup M
                                                                S : AddSubgroup M
                                                                m : M
                                                                ⊢ LE.le (Norm.norm (S.normedMk m)) (HMul.hMul 1 (Norm.norm m))
                                                              -/
  NormedAddGroupHom.opNorm_le_bound _ zero_le_one fun m => by simp [quotient_norm_mk_le']
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem _root_.QuotientAddGroup.norm_lift_apply_le {S : AddSubgroup M} (f : NormedAddGroupHom M N)
    (hf : ∀ x ∈ S, f x = 0) (x : M ⧸ S) : ‖lift S f.toAddMonoidHom hf x‖ ≤ ‖f‖ * ‖x‖ := by
  cases (norm_nonneg f).eq_or_gt with
  | inl h =>
    rcases mk_surjective x with ⟨x, rfl⟩
    simpa [h] using le_opNorm f x
  | inr h =>
    rw [← not_lt, ← lt_div_iff₀' h, norm_lt_iff]
    rintro ⟨x, rfl, hx⟩
    exact ((lt_div_iff₀' h).1 hx).not_le (le_opNorm f x)


/-- The operator norm of the projection is `1` if the subspace is not dense. -/
theorem norm_normedMk (S : AddSubgroup M) (h : (S.topologicalClosure : Set M) ≠ univ) :
    ‖S.normedMk‖ = 1 := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    h : Ne (↑S.topologicalClosure) Set.univ
    ⊢ Eq (Norm.norm S.normedMk) 1
  -/
  refine le_antisymm (norm_normedMk_le S) ?_
  obtain ⟨x, hx⟩ : ∃ x : M, 0 < ‖(x : M ⧸ S)‖ := by
    refine (Set.nonempty_compl.2 h).imp fun x hx ↦ ?_
    exact (norm_nonneg _).lt_of_ne' <| mt (quotient_norm_eq_zero_iff S x).1 hx
  /-
    case intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    h : Ne (↑S.topologicalClosure) Set.univ
    x : M
    hx : LT.lt 0 (Norm.norm ↑x)
    ⊢ LE.le 1 (Norm.norm S.normedMk)
  -/
  refine (le_mul_iff_one_le_left hx).1 ?_
  /-
    case intro
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    h : Ne (↑S.topologicalClosure) Set.univ
    x : M
    hx : LT.lt 0 (Norm.norm ↑x)
    ⊢ LE.le (Norm.norm ↑x) (HMul.hMul (Norm.norm S.normedMk) (Norm.norm ↑x))
  -/
  exact norm_lift_apply_le S.normedMk (fun x ↦ (eq_zero_iff x).2) x
  /-
    🎉 no goals
  -/


/-- The operator norm of the projection is `0` if the subspace is dense. -/
theorem norm_trivial_quotient_mk (S : AddSubgroup M)
    (h : (S.topologicalClosure : Set M) = Set.univ) : ‖S.normedMk‖ = 0 := by
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    h : Eq (↑S.topologicalClosure) Set.univ
    ⊢ Eq (Norm.norm S.normedMk) 0
  -/
  refine le_antisymm (opNorm_le_bound _ le_rfl fun x => ?_) (norm_nonneg _)
  have hker : x ∈ S.normedMk.ker.topologicalClosure := by
    rw [S.ker_normedMk, ← SetLike.mem_coe, h]
    trivial
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    h : Eq (↑S.topologicalClosure) Set.univ
    x : M
    hker : Membership.mem S.normedMk.ker.topologicalClosure x
    ⊢ LE.le (Norm.norm (S.normedMk x)) (HMul.hMul 0 (Norm.norm x))
  -/
  rw [ker_normedMk] at hker
  /-
    M : Type u_1
    inst✝ : SeminormedAddCommGroup M
    S : AddSubgroup M
    h : Eq (↑S.topologicalClosure) Set.univ
    x : M
    hker : Membership.mem S.topologicalClosure x
    ⊢ LE.le (Norm.norm (S.normedMk x)) (HMul.hMul 0 (Norm.norm x))
  -/
  simp only [(quotient_norm_eq_zero_iff S x).mpr hker, normedMk.apply, zero_mul, le_rfl]
  /-
    🎉 no goals
  -/


/-- `IsQuotient f`, for `f : M ⟶ N` means that `N` is isomorphic to the quotient of `M`
by the kernel of `f`. -/
structure IsQuotient (f : NormedAddGroupHom M N) : Prop where
  protected surjective : Function.Surjective f
  protected norm : ∀ x, ‖f x‖ = sInf ((fun m => ‖x + m‖) '' f.ker)


/-- Given `f : NormedAddGroupHom M N` such that `f s = 0` for all `s ∈ S`, where,
`S : AddSubgroup M` is closed, the induced morphism `NormedAddGroupHom (M ⧸ S) N`. -/
noncomputable def lift {N : Type*} [SeminormedAddCommGroup N] (S : AddSubgroup M)
    (f : NormedAddGroupHom M N) (hf : ∀ s ∈ S, f s = 0) : NormedAddGroupHom (M ⧸ S) N :=
  { QuotientAddGroup.lift S f.toAddMonoidHom hf with
    bound' := ⟨‖f‖, norm_lift_apply_le f hf⟩ }


theorem lift_mk {N : Type*} [SeminormedAddCommGroup N] (S : AddSubgroup M)
    (f : NormedAddGroupHom M N) (hf : ∀ s ∈ S, f s = 0) (m : M) :
    lift S f hf (S.normedMk m) = f m :=
  rfl


theorem lift_unique {N : Type*} [SeminormedAddCommGroup N] (S : AddSubgroup M)
    (f : NormedAddGroupHom M N) (hf : ∀ s ∈ S, f s = 0) (g : NormedAddGroupHom (M ⧸ S) N)
    (h : g.comp S.normedMk = f) : g = lift S f hf := by
  /-
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    g : NormedAddGroupHom (HasQuotient.Quotient M S) N
    h : Eq (g.comp S.normedMk) f
    ⊢ Eq g (NormedAddGroupHom.lift S f hf)
  -/
  ext x
  /-
    case H
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    g : NormedAddGroupHom (HasQuotient.Quotient M S) N
    h : Eq (g.comp S.normedMk) f
    x : HasQuotient.Quotient M S
    ⊢ Eq (g x) ((NormedAddGroupHom.lift S f hf) x)
  -/
  rcases AddSubgroup.surjective_normedMk _ x with ⟨x, rfl⟩
  /-
    case H.intro
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    g : NormedAddGroupHom (HasQuotient.Quotient M S) N
    h : Eq (g.comp S.normedMk) f
    x : M
    ⊢ Eq (g (S.normedMk x)) ((NormedAddGroupHom.lift S f hf) (S.normedMk x))
  -/
  change g.comp S.normedMk x = _
  /-
    case H.intro
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    g : NormedAddGroupHom (HasQuotient.Quotient M S) N
    h : Eq (g.comp S.normedMk) f
    x : M
    ⊢ Eq ((g.comp S.normedMk) x) ((NormedAddGroupHom.lift S f hf) (S.normedMk x))
  -/
  simp only [h]
  /-
    case H.intro
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    g : NormedAddGroupHom (HasQuotient.Quotient M S) N
    h : Eq (g.comp S.normedMk) f
    x : M
    ⊢ Eq (f x) ((NormedAddGroupHom.lift S f hf) (S.normedMk x))
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `S.normedMk` satisfies `IsQuotient`. -/
theorem isQuotientQuotient (S : AddSubgroup M) : IsQuotient S.normedMk :=
                                      /-
                                        M : Type u_1
                                        inst✝ : SeminormedAddCommGroup M
                                        S : AddSubgroup M
                                        m : M
                                        ⊢ Eq (Norm.norm (S.normedMk m)) (InfSet.sInf (Set.image (fun m_1 => Norm.norm  …
                                      -/
  ⟨S.surjective_normedMk, fun m => by simpa [S.ker_normedMk] using quotient_norm_mk_eq _ m⟩
                                      /-
                                        🎉 no goals
                                      -/


theorem IsQuotient.norm_lift {f : NormedAddGroupHom M N} (hquot : IsQuotient f) {ε : ℝ} (hε : 0 < ε)
    (n : N) : ∃ m : M, f m = n ∧ ‖m‖ < ‖n‖ + ε := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : SeminormedAddCommGroup M
    inst✝ : SeminormedAddCommGroup N
    f : NormedAddGroupHom M N
    hquot : f.IsQuotient
    ε : Real
    hε : LT.lt 0 ε
    n : N
    ⊢ Exists fun m => And (Eq (f m) n) (LT.lt (Norm.norm m) (HAdd.hAdd (Norm.norm  …
  -/
  obtain ⟨m, rfl⟩ := hquot.surjective n
  have nonemp : ((fun m' => ‖m + m'‖) '' f.ker).Nonempty := by
    rw [Set.image_nonempty]
    exact ⟨0, f.ker.zero_mem⟩
  rcases Real.lt_sInf_add_pos nonemp hε
    with ⟨_, ⟨⟨x, hx, rfl⟩, H : ‖m + x‖ < sInf ((fun m' : M => ‖m + m'‖) '' f.ker) + ε⟩⟩
  exact ⟨m + x, by rw [map_add, (NormedAddGroupHom.mem_ker f x).mp hx, add_zero], by
    rwa [hquot.norm]⟩


theorem IsQuotient.norm_le {f : NormedAddGroupHom M N} (hquot : IsQuotient f) (m : M) :
    ‖f m‖ ≤ ‖m‖ := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : SeminormedAddCommGroup M
    inst✝ : SeminormedAddCommGroup N
    f : NormedAddGroupHom M N
    hquot : f.IsQuotient
    m : M
    ⊢ LE.le (Norm.norm (f m)) (Norm.norm m)
  -/
  rw [hquot.norm]
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : SeminormedAddCommGroup M
    inst✝ : SeminormedAddCommGroup N
    f : NormedAddGroupHom M N
    hquot : f.IsQuotient
    m : M
    ⊢ LE.le (InfSet.sInf (Set.image (fun m_1 => Norm.norm (HAdd.hAdd m m_1)) ↑f.ke …
  -/
  apply csInf_le
    /-
      case h₁
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      f : NormedAddGroupHom M N
      hquot : f.IsQuotient
      m : M
      ⊢ BddBelow (Set.image (fun m_1 => Norm.norm (HAdd.hAdd m m_1)) ↑f.ker)
    -/
  · use 0
    /-
      case h
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      f : NormedAddGroupHom M N
      hquot : f.IsQuotient
      m : M
      ⊢ Membership.mem (lowerBounds (Set.image (fun m_1 => Norm.norm (HAdd.hAdd m m_ …
    -/
    rintro _ ⟨m', -, rfl⟩
    /-
      case h.intro.intro
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      f : NormedAddGroupHom M N
      hquot : f.IsQuotient
      m m' : M
      ⊢ LE.le 0 ((fun m_1 => Norm.norm (HAdd.hAdd m m_1)) m')
    -/
    apply norm_nonneg
    /-
      🎉 no goals
    -/
    /-
      case h₂
      M : Type u_1
      N : Type u_2
      inst✝¹ : SeminormedAddCommGroup M
      inst✝ : SeminormedAddCommGroup N
      f : NormedAddGroupHom M N
      hquot : f.IsQuotient
      m : M
      ⊢ Membership.mem (Set.image (fun m_1 => Norm.norm (HAdd.hAdd m m_1)) ↑f.ker) ( …
    -/
  · exact ⟨0, f.ker.zero_mem, by simp⟩
    /-
      🎉 no goals
    -/


theorem norm_lift_le {N : Type*} [SeminormedAddCommGroup N] (S : AddSubgroup M)
    (f : NormedAddGroupHom M N) (hf : ∀ s ∈ S, f s = 0) :
    ‖lift S f hf‖ ≤ ‖f‖ :=
  opNorm_le_bound _ (norm_nonneg f) (norm_lift_apply_le f hf)

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: deprecate?

theorem lift_norm_le {N : Type*} [SeminormedAddCommGroup N] (S : AddSubgroup M)
    (f : NormedAddGroupHom M N) (hf : ∀ s ∈ S, f s = 0) {c : ℝ≥0} (fb : ‖f‖ ≤ c) :
    ‖lift S f hf‖ ≤ c :=
  (norm_lift_le S f hf).trans fb


theorem lift_normNoninc {N : Type*} [SeminormedAddCommGroup N] (S : AddSubgroup M)
    (f : NormedAddGroupHom M N) (hf : ∀ s ∈ S, f s = 0) (fb : f.NormNoninc) :
    (lift S f hf).NormNoninc := fun x => by
  /-
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    fb : f.NormNoninc
    x : HasQuotient.Quotient M S
    ⊢ LE.le (Norm.norm ((NormedAddGroupHom.lift S f hf) x)) (Norm.norm x)
  -/
  have fb' : ‖f‖ ≤ (1 : ℝ≥0) := NormNoninc.normNoninc_iff_norm_le_one.mp fb
  /-
    M : Type u_1
    inst✝¹ : SeminormedAddCommGroup M
    N : Type u_3
    inst✝ : SeminormedAddCommGroup N
    S : AddSubgroup M
    f : NormedAddGroupHom M N
    hf : ∀ (s : M), Membership.mem S s → Eq (f s) 0
    fb : f.NormNoninc
    x : HasQuotient.Quotient M S
    fb' : LE.le (Norm.norm f) ↑1
    ⊢ LE.le (Norm.norm ((NormedAddGroupHom.lift S f hf) x)) (Norm.norm x)
  -/
  simpa using le_of_opNorm_le _ (f.lift_norm_le _ _ fb') _
  /-
    🎉 no goals
  -/


instance Submodule.Quotient.seminormedAddCommGroup : SeminormedAddCommGroup (M ⧸ S) :=
  AddSubgroup.seminormedAddCommGroupQuotient S.toAddSubgroup


instance Submodule.Quotient.normedAddCommGroup [hS : IsClosed (S : Set M)] :
    NormedAddCommGroup (M ⧸ S) :=
  @AddSubgroup.normedAddCommGroupQuotient _ _ S.toAddSubgroup hS


instance Submodule.Quotient.completeSpace [CompleteSpace M] : CompleteSpace (M ⧸ S) :=
  QuotientAddGroup.completeSpace M S.toAddSubgroup


/-- For any `x : M ⧸ S` and any `0 < ε`, there is `m : M` such that `Submodule.Quotient.mk m = x`
and `‖m‖ < ‖x‖ + ε`. -/
nonrec theorem Submodule.Quotient.norm_mk_lt {S : Submodule R M} (x : M ⧸ S) {ε : ℝ} (hε : 0 < ε) :
    ∃ m : M, Submodule.Quotient.mk m = x ∧ ‖m‖ < ‖x‖ + ε :=
  norm_mk_lt x hε


theorem Submodule.Quotient.norm_mk_le (m : M) : ‖(Submodule.Quotient.mk m : M ⧸ S)‖ ≤ ‖m‖ :=
  quotient_norm_mk_le S.toAddSubgroup m


instance Submodule.Quotient.instBoundedSMul (𝕜 : Type*)
    [SeminormedCommRing 𝕜] [Module 𝕜 M] [BoundedSMul 𝕜 M] [SMul 𝕜 R] [IsScalarTower 𝕜 R M] :
    BoundedSMul 𝕜 (M ⧸ S) :=
  .of_norm_smul_le fun k x =>
    -- Porting note: this is `QuotientAddGroup.norm_lift_apply_le` for `f : M → M ⧸ S` given by
    -- `x ↦ mk (k • x)`; todo: add scalar multiplication as `NormedAddGroupHom`, use it here
    _root_.le_of_forall_pos_le_add fun ε hε => by
      have := (nhds_basis_ball.tendsto_iff nhds_basis_ball).mp
        ((@Real.uniformContinuous_const_mul ‖k‖).continuous.tendsto ‖x‖) ε hε
      /-
        M : Type u_1
        N : Type u_2
        inst✝⁸ : SeminormedAddCommGroup M
        inst✝⁷ : SeminormedAddCommGroup N
        R : Type u_3
        inst✝⁶ : Ring R
        inst✝⁵ : Module R M
        S : Submodule R M
        𝕜 : Type u_4
        inst✝⁴ : SeminormedCommRing 𝕜
        inst✝³ : Module 𝕜 M
        inst✝² : BoundedSMul 𝕜 M
        inst✝¹ : SMul 𝕜 R
        inst✝ : IsScalarTower 𝕜 R M
        k : 𝕜
        x : HasQuotient.Quotient M S
        ε : Real
        hε : LT.lt 0 ε
        this : Exists fun ia => And (LT.lt 0 ia) (∀ (x_1 : Real), Membership.mem (Metr …
        ⊢ LE.le (Norm.norm (HSMul.hSMul k x)) (HAdd.hAdd (HMul.hMul (Norm.norm k) (Nor …
      -/
      simp only [mem_ball, exists_prop, dist, abs_sub_lt_iff] at this
      /-
        M : Type u_1
        N : Type u_2
        inst✝⁸ : SeminormedAddCommGroup M
        inst✝⁷ : SeminormedAddCommGroup N
        R : Type u_3
        inst✝⁶ : Ring R
        inst✝⁵ : Module R M
        S : Submodule R M
        𝕜 : Type u_4
        inst✝⁴ : SeminormedCommRing 𝕜
        inst✝³ : Module 𝕜 M
        inst✝² : BoundedSMul 𝕜 M
        inst✝¹ : SMul 𝕜 R
        inst✝ : IsScalarTower 𝕜 R M
        k : 𝕜
        x : HasQuotient.Quotient M S
        ε : Real
        hε : LT.lt 0 ε
        this : Exists fun ia => And (LT.lt 0 ia) (∀ (x_1 : Real), And (LT.lt (HSub.hSu …
        ⊢ LE.le (Norm.norm (HSMul.hSMul k x)) (HAdd.hAdd (HMul.hMul (Norm.norm k) (Nor …
      -/
      rcases this with ⟨δ, hδ, h⟩
      /-
        case intro.intro
        M : Type u_1
        N : Type u_2
        inst✝⁸ : SeminormedAddCommGroup M
        inst✝⁷ : SeminormedAddCommGroup N
        R : Type u_3
        inst✝⁶ : Ring R
        inst✝⁵ : Module R M
        S : Submodule R M
        𝕜 : Type u_4
        inst✝⁴ : SeminormedCommRing 𝕜
        inst✝³ : Module 𝕜 M
        inst✝² : BoundedSMul 𝕜 M
        inst✝¹ : SMul 𝕜 R
        inst✝ : IsScalarTower 𝕜 R M
        k : 𝕜
        x : HasQuotient.Quotient M S
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ : LT.lt 0 δ
        h : ∀ (x_1 : Real), And (LT.lt (HSub.hSub x_1 (Norm.norm x)) δ) (LT.lt (HSub.h …
        ⊢ LE.le (Norm.norm (HSMul.hSMul k x)) (HAdd.hAdd (HMul.hMul (Norm.norm k) (Nor …
      -/
      obtain ⟨a, rfl, ha⟩ := Submodule.Quotient.norm_mk_lt x hδ
      /-
        case intro.intro.intro.intro
        M : Type u_1
        N : Type u_2
        inst✝⁸ : SeminormedAddCommGroup M
        inst✝⁷ : SeminormedAddCommGroup N
        R : Type u_3
        inst✝⁶ : Ring R
        inst✝⁵ : Module R M
        S : Submodule R M
        𝕜 : Type u_4
        inst✝⁴ : SeminormedCommRing 𝕜
        inst✝³ : Module 𝕜 M
        inst✝² : BoundedSMul 𝕜 M
        inst✝¹ : SMul 𝕜 R
        inst✝ : IsScalarTower 𝕜 R M
        k : 𝕜
        ε : Real
        hε : LT.lt 0 ε
        δ : Real
        hδ : LT.lt 0 δ
        a : M
        h : ∀ (x : Real), And (LT.lt (HSub.hSub x (Norm.norm (Submodule.Quotient.mk a) …
        ha : LT.lt (Norm.norm a) (HAdd.hAdd (Norm.norm (Submodule.Quotient.mk a)) δ)
        ⊢ LE.le (Norm.norm (HSMul.hSMul k (Submodule.Quotient.mk a))) (HAdd.hAdd (HMul …
      -/
      specialize h ‖a‖ ⟨by linarith, by linarith [Submodule.Quotient.norm_mk_le S a]⟩
      calc
        _ ≤ ‖k‖ * ‖a‖ := (quotient_norm_mk_le S.toAddSubgroup (k • a)).trans (norm_smul_le k a)
        _ ≤ _ := (sub_lt_iff_lt_add'.mp h.1).le


instance Submodule.Quotient.normedSpace (𝕜 : Type*) [NormedField 𝕜] [NormedSpace 𝕜 M] [SMul 𝕜 R]
    [IsScalarTower 𝕜 R M] : NormedSpace 𝕜 (M ⧸ S) where
  norm_smul_le := norm_smul_le


nonrec theorem Ideal.Quotient.norm_mk_lt {I : Ideal R} (x : R ⧸ I) {ε : ℝ} (hε : 0 < ε) :
    ∃ r : R, Ideal.Quotient.mk I r = x ∧ ‖r‖ < ‖x‖ + ε :=
  norm_mk_lt x hε


theorem Ideal.Quotient.norm_mk_le (r : R) : ‖Ideal.Quotient.mk I r‖ ≤ ‖r‖ :=
  quotient_norm_mk_le I.toAddSubgroup r


instance Ideal.Quotient.semiNormedCommRing : SeminormedCommRing (R ⧸ I) where
  dist_eq := dist_eq_norm
  mul_comm := _root_.mul_comm
  norm_mul x y := le_of_forall_pos_le_add fun ε hε => by
    have := ((nhds_basis_ball.prod_nhds nhds_basis_ball).tendsto_iff nhds_basis_ball).mp
      (continuous_mul.tendsto (‖x‖, ‖y‖)) ε hε
    /-
      M : Type u_1
      N : Type u_2
      inst✝² : SeminormedAddCommGroup M
      inst✝¹ : SeminormedAddCommGroup N
      R : Type u_3
      inst✝ : SeminormedCommRing R
      I : Ideal R
      x y : HasQuotient.Quotient R I
      ε : Real
      hε : LT.lt 0 ε
      this : Exists fun ia => And (And (LT.lt 0 ia.1) (LT.lt 0 ia.2)) (∀ (x_1 : Prod …
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm. …
    -/
    simp only [Set.mem_prod, mem_ball, and_imp, Prod.forall, exists_prop, Prod.exists] at this
    /-
      M : Type u_1
      N : Type u_2
      inst✝² : SeminormedAddCommGroup M
      inst✝¹ : SeminormedAddCommGroup N
      R : Type u_3
      inst✝ : SeminormedCommRing R
      I : Ideal R
      x y : HasQuotient.Quotient R I
      ε : Real
      hε : LT.lt 0 ε
      this : Exists fun a => Exists fun b => And (And (LT.lt 0 a) (LT.lt 0 b)) (∀ (a …
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm. …
    -/
    rcases this with ⟨ε₁, ε₂, ⟨h₁, h₂⟩, h⟩
    obtain ⟨⟨a, rfl, ha⟩, ⟨b, rfl, hb⟩⟩ := Ideal.Quotient.norm_mk_lt x h₁,
      Ideal.Quotient.norm_mk_lt y h₂
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      M : Type u_1
      N : Type u_2
      inst✝² : SeminormedAddCommGroup M
      inst✝¹ : SeminormedAddCommGroup N
      R : Type u_3
      inst✝ : SeminormedCommRing R
      I : Ideal R
      ε : Real
      hε : LT.lt 0 ε
      ε₁ ε₂ : Real
      h₁ : LT.lt 0 ε₁
      h₂ : LT.lt 0 ε₂
      a : R
      ha : LT.lt (Norm.norm a) (HAdd.hAdd (Norm.norm ((Ideal.Quotient.mk I) a)) ε₁)
      b : R
      h : ∀ (a_1 b_1 : Real), LT.lt (Dist.dist a_1 (Norm.norm ((Ideal.Quotient.mk I) …
      hb : LT.lt (Norm.norm b) (HAdd.hAdd (Norm.norm ((Ideal.Quotient.mk I) b)) ε₂)
      ⊢ LE.le (Norm.norm (HMul.hMul ((Ideal.Quotient.mk I) a) ((Ideal.Quotient.mk I) …
    -/
    simp only [dist, abs_sub_lt_iff] at h
    specialize h ‖a‖ ‖b‖ ⟨by linarith, by linarith [Ideal.Quotient.norm_mk_le I a]⟩
      ⟨by linarith, by linarith [Ideal.Quotient.norm_mk_le I b]⟩
    calc
      _ ≤ ‖a‖ * ‖b‖ := (Ideal.Quotient.norm_mk_le I (a * b)).trans (norm_mul_le a b)
      _ ≤ _ := (sub_lt_iff_lt_add'.mp h.1).le


instance Ideal.Quotient.normedCommRing [IsClosed (I : Set R)] : NormedCommRing (R ⧸ I) :=
  { Ideal.Quotient.semiNormedCommRing I, Submodule.Quotient.normedAddCommGroup I with }


instance Ideal.Quotient.normedAlgebra [NormedAlgebra 𝕜 R] : NormedAlgebra 𝕜 (R ⧸ I) :=
  { Submodule.Quotient.normedSpace I 𝕜, Ideal.Quotient.algebra 𝕜 with }


