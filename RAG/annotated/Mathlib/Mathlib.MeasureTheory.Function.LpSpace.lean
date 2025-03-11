@[simp]
theorem eLpNorm_aeeqFun {α E : Type*} [MeasurableSpace α] {μ : Measure α} [NormedAddCommGroup E]
    {p : ℝ≥0∞} {f : α → E} (hf : AEStronglyMeasurable f μ) :
    eLpNorm (AEEqFun.mk f hf) p μ = eLpNorm f p μ :=
  eLpNorm_congr_ae (AEEqFun.coeFn_mk _ _)


@[deprecated (since := "2024-07-27")]
alias snorm_aeeqFun := eLpNorm_aeeqFun


theorem Memℒp.eLpNorm_mk_lt_top {α E : Type*} [MeasurableSpace α] {μ : Measure α}
    [NormedAddCommGroup E] {p : ℝ≥0∞} {f : α → E} (hfp : Memℒp f p μ) :
                                               /-
                                                 α : Type u_5
                                                 E : Type u_6
                                                 inst✝¹ : MeasurableSpace α
                                                 μ : MeasureTheory.Measure α
                                                 inst✝ : NormedAddCommGroup E
                                                 p : ENNReal
                                                 f : α → E
                                                 hfp : MeasureTheory.Memℒp f p μ
                                                 ⊢ LT.lt (MeasureTheory.eLpNorm (↑(MeasureTheory.AEEqFun.mk f ⋯)) p μ) Top.top
                                               -/
    eLpNorm (AEEqFun.mk f hfp.1) p μ < ∞ := by simp [hfp.2]
                                               /-
                                                 🎉 no goals
                                               -/


@[deprecated (since := "2024-07-27")]
alias Memℒp.snorm_mk_lt_top := Memℒp.eLpNorm_mk_lt_top


/-- Lp space -/
def Lp {α} (E : Type*) {m : MeasurableSpace α} [NormedAddCommGroup E] (p : ℝ≥0∞)
    (μ : Measure α := by volume_tac) : AddSubgroup (α →ₘ[μ] E) where
  carrier := { f | eLpNorm f p μ < ∞ }
                  /-
                    α✝ : Type u_1
                    E✝ : Type u_2
                    F : Type u_3
                    G : Type u_4
                    m✝ m0 : MeasurableSpace α✝
                    p✝ : ENNReal
                    q : Real
                    μ✝ ν : MeasureTheory.Measure α✝
                    inst✝³ : NormedAddCommGroup E✝
                    inst✝² : NormedAddCommGroup F
                    inst✝¹ : NormedAddCommGroup G
                    α : Type ?u.3975
                    E : Type u_5
                    m : MeasurableSpace α
                    inst✝ : NormedAddCommGroup E
                    p : ENNReal
                    μ : autoParam (MeasureTheory.Measure α) _auto✝
                    ⊢ Membership.mem { carrier := setOf fun f => LT.lt (MeasureTheory.eLpNorm (↑f) …
                  -/
  zero_mem' := by simp [eLpNorm_congr_ae AEEqFun.coeFn_zero, eLpNorm_zero]
                  /-
                    🎉 no goals
                  -/
  add_mem' {f g} hf hg := by
    simp [eLpNorm_congr_ae (AEEqFun.coeFn_add f g),
      eLpNorm_add_lt_top ⟨f.aestronglyMeasurable, hf⟩ ⟨g.aestronglyMeasurable, hg⟩]
                        /-
                          α✝ : Type u_1
                          E✝ : Type u_2
                          F : Type u_3
                          G : Type u_4
                          m✝ m0 : MeasurableSpace α✝
                          p✝ : ENNReal
                          q : Real
                          μ✝ ν : MeasureTheory.Measure α✝
                          inst✝³ : NormedAddCommGroup E✝
                          inst✝² : NormedAddCommGroup F
                          inst✝¹ : NormedAddCommGroup G
                          α : Type ?u.3975
                          E : Type u_5
                          m : MeasurableSpace α
                          inst✝ : NormedAddCommGroup E
                          p : ENNReal
                          μ : autoParam (MeasureTheory.Measure α) _auto✝
                          f : MeasureTheory.AEEqFun α E μ
                          hf : Membership.mem { carrier := setOf fun f => LT.lt (MeasureTheory.eLpNorm ( …
                          ⊢ Membership.mem { carrier := setOf fun f => LT.lt (MeasureTheory.eLpNorm (↑f) …
                        -/
  neg_mem' {f} hf := by rwa [Set.mem_setOf_eq, eLpNorm_congr_ae (AEEqFun.coeFn_neg f), eLpNorm_neg]
                        /-
                          🎉 no goals
                        -/

-- Porting note: calling the first argument `α` breaks the `(α := ·)` notation

scoped notation:25 α' " →₁[" μ "] " E => MeasureTheory.Lp (α := α') E 1 μ

scoped notation:25 α' " →₂[" μ "] " E => MeasureTheory.Lp (α := α') E 2 μ


/-- make an element of Lp from a function verifying `Memℒp` -/
def toLp (f : α → E) (h_mem_ℒp : Memℒp f p μ) : Lp E p μ :=
  ⟨AEEqFun.mk f h_mem_ℒp.1, h_mem_ℒp.eLpNorm_mk_lt_top⟩


theorem toLp_val {f : α → E} (h : Memℒp f p μ) : (toLp f h).1 = AEEqFun.mk f h.1 := rfl


theorem coeFn_toLp {f : α → E} (hf : Memℒp f p μ) : hf.toLp f =ᵐ[μ] f :=
  AEEqFun.coeFn_mk _ _


theorem toLp_congr {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) (hfg : f =ᵐ[μ] g) :
                                /-
                                  α : Type u_1
                                  E : Type u_2
                                  m0 : MeasurableSpace α
                                  p : ENNReal
                                  μ : MeasureTheory.Measure α
                                  inst✝ : NormedAddCommGroup E
                                  f g : α → E
                                  hf : MeasureTheory.Memℒp f p μ
                                  hg : MeasureTheory.Memℒp g p μ
                                  hfg : (MeasureTheory.ae μ).EventuallyEq f g
                                  ⊢ Eq (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg)
                                -/
    hf.toLp f = hg.toLp g := by simp [toLp, hfg]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem toLp_eq_toLp_iff {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
                                            /-
                                              α : Type u_1
                                              E : Type u_2
                                              m0 : MeasurableSpace α
                                              p : ENNReal
                                              μ : MeasureTheory.Measure α
                                              inst✝ : NormedAddCommGroup E
                                              f g : α → E
                                              hf : MeasureTheory.Memℒp f p μ
                                              hg : MeasureTheory.Memℒp g p μ
                                              ⊢ Iff (Eq (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g hg)) ((M …
                                            -/
    hf.toLp f = hg.toLp g ↔ f =ᵐ[μ] g := by simp [toLp]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem toLp_zero (h : Memℒp (0 : α → E) p μ) : h.toLp 0 = 0 :=
  rfl


theorem toLp_add {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    (hf.add hg).toLp (f + g) = hf.toLp f + hg.toLp g :=
  rfl


theorem toLp_neg {f : α → E} (hf : Memℒp f p μ) : hf.neg.toLp (-f) = -hf.toLp f :=
  rfl


theorem toLp_sub {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    (hf.sub hg).toLp (f - g) = hf.toLp f - hg.toLp g :=
  rfl


instance instCoeFun : CoeFun (Lp E p μ) (fun _ => α → E) :=
  ⟨fun f => ((f : α →ₘ[μ] E) : α → E)⟩


@[ext high]
theorem ext {f g : Lp E p μ} (h : f =ᵐ[μ] g) : f = g := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑f ↑↑g
    ⊢ Eq f g
  -/
  cases f
  /-
    case mk
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    val✝ : MeasureTheory.AEEqFun α E μ
    property✝ : Membership.mem (MeasureTheory.Lp E p μ) val✝
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑⟨val✝, property✝⟩ ↑↑g
    ⊢ Eq ⟨val✝, property✝⟩ g
  -/
  cases g
  /-
    case mk.mk
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    val✝¹ : MeasureTheory.AEEqFun α E μ
    property✝¹ : Membership.mem (MeasureTheory.Lp E p μ) val✝¹
    val✝ : MeasureTheory.AEEqFun α E μ
    property✝ : Membership.mem (MeasureTheory.Lp E p μ) val✝
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑⟨val✝¹, property✝¹⟩ ↑↑⟨val✝, property✝⟩
    ⊢ Eq ⟨val✝¹, property✝¹⟩ ⟨val✝, property✝⟩
  -/
  simp only [Subtype.mk_eq_mk]
  /-
    case mk.mk
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    val✝¹ : MeasureTheory.AEEqFun α E μ
    property✝¹ : Membership.mem (MeasureTheory.Lp E p μ) val✝¹
    val✝ : MeasureTheory.AEEqFun α E μ
    property✝ : Membership.mem (MeasureTheory.Lp E p μ) val✝
    h : (MeasureTheory.ae μ).EventuallyEq ↑↑⟨val✝¹, property✝¹⟩ ↑↑⟨val✝, property✝⟩
    ⊢ Eq val✝¹ val✝
  -/
  exact AEEqFun.ext h
  /-
    🎉 no goals
  -/


theorem mem_Lp_iff_eLpNorm_lt_top {f : α →ₘ[μ] E} : f ∈ Lp E p μ ↔ eLpNorm f p μ < ∞ := Iff.rfl


@[deprecated (since := "2024-07-27")]
alias mem_Lp_iff_snorm_lt_top := mem_Lp_iff_eLpNorm_lt_top


theorem mem_Lp_iff_memℒp {f : α →ₘ[μ] E} : f ∈ Lp E p μ ↔ Memℒp f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : MeasureTheory.AEEqFun α E μ
    ⊢ Iff (Membership.mem (MeasureTheory.Lp E p μ) f) (MeasureTheory.Memℒp (↑f) p μ)
  -/
  simp [mem_Lp_iff_eLpNorm_lt_top, Memℒp, f.stronglyMeasurable.aestronglyMeasurable]
  /-
    🎉 no goals
  -/


protected theorem antitone [IsFiniteMeasure μ] {p q : ℝ≥0∞} (hpq : p ≤ q) : Lp E q μ ≤ Lp E p μ :=
  fun f hf => (Memℒp.memℒp_of_exponent_le ⟨f.aestronglyMeasurable, hf⟩ hpq).2


@[simp]
theorem coeFn_mk {f : α →ₘ[μ] E} (hf : eLpNorm f p μ < ∞) : ((⟨f, hf⟩ : Lp E p μ) : α → E) = f :=
  rfl

-- @[simp] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10685): dsimp can prove this

theorem coe_mk {f : α →ₘ[μ] E} (hf : eLpNorm f p μ < ∞) : ((⟨f, hf⟩ : Lp E p μ) : α →ₘ[μ] E) = f :=
  rfl


@[simp]
theorem toLp_coeFn (f : Lp E p μ) (hf : Memℒp f p μ) : hf.toLp f = f := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hf : MeasureTheory.Memℒp (↑↑f) p μ
    ⊢ Eq (MeasureTheory.Memℒp.toLp (↑↑f) hf) f
  -/
  cases f
  /-
    case mk
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    val✝ : MeasureTheory.AEEqFun α E μ
    property✝ : Membership.mem (MeasureTheory.Lp E p μ) val✝
    hf : MeasureTheory.Memℒp (↑↑⟨val✝, property✝⟩) p μ
    ⊢ Eq (MeasureTheory.Memℒp.toLp (↑↑⟨val✝, property✝⟩) hf) ⟨val✝, property✝⟩
  -/
  simp [Memℒp.toLp]
  /-
    🎉 no goals
  -/


theorem eLpNorm_lt_top (f : Lp E p μ) : eLpNorm f p μ < ∞ :=
  f.prop


@[deprecated (since := "2024-07-27")]
alias snorm_lt_top := eLpNorm_lt_top


theorem eLpNorm_ne_top (f : Lp E p μ) : eLpNorm f p μ ≠ ∞ :=
  (eLpNorm_lt_top f).ne


@[deprecated (since := "2024-07-27")]
alias snorm_ne_top := eLpNorm_ne_top


@[measurability]
protected theorem stronglyMeasurable (f : Lp E p μ) : StronglyMeasurable f :=
  f.val.stronglyMeasurable


@[measurability]
protected theorem aestronglyMeasurable (f : Lp E p μ) : AEStronglyMeasurable f μ :=
  f.val.aestronglyMeasurable


protected theorem memℒp (f : Lp E p μ) : Memℒp f p μ :=
  ⟨Lp.aestronglyMeasurable f, f.prop⟩


theorem coeFn_zero : ⇑(0 : Lp E p μ) =ᵐ[μ] 0 :=
  AEEqFun.coeFn_zero


theorem coeFn_neg (f : Lp E p μ) : ⇑(-f) =ᵐ[μ] -f :=
  AEEqFun.coeFn_neg _


theorem coeFn_add (f g : Lp E p μ) : ⇑(f + g) =ᵐ[μ] f + g :=
  AEEqFun.coeFn_add _ _


theorem coeFn_sub (f g : Lp E p μ) : ⇑(f - g) =ᵐ[μ] f - g :=
  AEEqFun.coeFn_sub _ _


theorem const_mem_Lp (α) {_ : MeasurableSpace α} (μ : Measure α) (c : E) [IsFiniteMeasure μ] :
    @AEEqFun.const α _ _ μ _ c ∈ Lp E p μ :=
  (memℒp_const c).eLpNorm_mk_lt_top


instance instNorm : Norm (Lp E p μ) where norm f := ENNReal.toReal (eLpNorm f p μ)

-- note: we need this to be defeq to the instance from `SeminormedAddGroup.toNNNorm`, so
-- can't use `ENNReal.toNNReal (eLpNorm f p μ)`

instance instNNNorm : NNNorm (Lp E p μ) where nnnorm f := ⟨‖f‖, ENNReal.toReal_nonneg⟩


instance instDist : Dist (Lp E p μ) where dist f g := ‖f - g‖


instance instEDist : EDist (Lp E p μ) where edist f g := eLpNorm (⇑f - ⇑g) p μ


theorem norm_def (f : Lp E p μ) : ‖f‖ = ENNReal.toReal (eLpNorm f p μ) :=
  rfl


theorem nnnorm_def (f : Lp E p μ) : ‖f‖₊ = ENNReal.toNNReal (eLpNorm f p μ) :=
  rfl


@[simp, norm_cast]
protected theorem coe_nnnorm (f : Lp E p μ) : (‖f‖₊ : ℝ) = ‖f‖ :=
  rfl


@[simp, norm_cast]
theorem nnnorm_coe_ennreal (f : Lp E p μ) : (‖f‖₊ : ℝ≥0∞) = eLpNorm f p μ :=
  ENNReal.coe_toNNReal <| Lp.eLpNorm_ne_top f


@[simp]
lemma norm_toLp (f : α → E) (hf : Memℒp f p μ) : ‖hf.toLp f‖ = ENNReal.toReal (eLpNorm f p μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ Eq (Norm.norm (MeasureTheory.Memℒp.toLp f hf)) (MeasureTheory.eLpNorm f p μ) …
  -/
  rw [norm_def, eLpNorm_congr_ae (Memℒp.coeFn_toLp hf)]
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_toLp (f : α → E) (hf : Memℒp f p μ) :
    ‖hf.toLp f‖₊ = ENNReal.toNNReal (eLpNorm f p μ) :=
  NNReal.eq <| norm_toLp f hf


theorem coe_nnnorm_toLp {f : α → E} (hf : Memℒp f p μ) : (‖hf.toLp f‖₊ : ℝ≥0∞) = eLpNorm f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ Eq (↑(NNNorm.nnnorm (MeasureTheory.Memℒp.toLp f hf))) (MeasureTheory.eLpNorm …
  -/
  rw [nnnorm_toLp f hf, ENNReal.coe_toNNReal hf.2.ne]
  /-
    🎉 no goals
  -/


theorem dist_def (f g : Lp E p μ) : dist f g = (eLpNorm (⇑f - ⇑g) p μ).toReal := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Eq (Dist.dist f g) (MeasureTheory.eLpNorm (HSub.hSub ↑↑f ↑↑g) p μ).toReal
  -/
  simp_rw [dist, norm_def]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Eq (MeasureTheory.eLpNorm (↑↑(HSub.hSub f g)) p μ).toReal (MeasureTheory.eLp …
  -/
  refine congr_arg _ ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Eq (MeasureTheory.eLpNorm (↑↑(HSub.hSub f g)) p μ) (MeasureTheory.eLpNorm (H …
  -/
  apply eLpNorm_congr_ae (coeFn_sub _ _)
  /-
    🎉 no goals
  -/


theorem edist_def (f g : Lp E p μ) : edist f g = eLpNorm (⇑f - ⇑g) p μ :=
  rfl


protected theorem edist_dist (f g : Lp E p μ) : edist f g = .ofReal (dist f g) := by
  rw [edist_def, dist_def, ← eLpNorm_congr_ae (coeFn_sub _ _),
    ENNReal.ofReal_toReal (eLpNorm_ne_top (f - g))]


protected theorem dist_edist (f g : Lp E p μ) : dist f g = (edist f g).toReal :=
  MeasureTheory.Lp.dist_def ..


theorem dist_eq_norm (f g : Lp E p μ) : dist f g = ‖f - g‖ := rfl


@[simp]
theorem edist_toLp_toLp (f g : α → E) (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    edist (hf.toLp f) (hg.toLp g) = eLpNorm (f - g) p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : α → E
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    ⊢ Eq (EDist.edist (MeasureTheory.Memℒp.toLp f hf) (MeasureTheory.Memℒp.toLp g  …
  -/
  rw [edist_def]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : α → E
    hf : MeasureTheory.Memℒp f p μ
    hg : MeasureTheory.Memℒp g p μ
    ⊢ Eq (MeasureTheory.eLpNorm (HSub.hSub ↑↑(MeasureTheory.Memℒp.toLp f hf) ↑↑(Me …
  -/
  exact eLpNorm_congr_ae (hf.coeFn_toLp.sub hg.coeFn_toLp)
  /-
    🎉 no goals
  -/


@[simp]
theorem edist_toLp_zero (f : α → E) (hf : Memℒp f p μ) : edist (hf.toLp f) 0 = eLpNorm f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ Eq (EDist.edist (MeasureTheory.Memℒp.toLp f hf) 0) (MeasureTheory.eLpNorm f  …
  -/
  convert edist_toLp_toLp f 0 hf zero_memℒp
  /-
    case h.e'_3.h.e'_5
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    ⊢ Eq f (HSub.hSub f 0)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_zero : ‖(0 : Lp E p μ)‖₊ = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    ⊢ Eq (NNNorm.nnnorm 0) 0
  -/
  rw [nnnorm_def]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    ⊢ Eq (MeasureTheory.eLpNorm (↑↑0) p μ).toNNReal 0
  -/
  change (eLpNorm (⇑(0 : α →ₘ[μ] E)) p μ).toNNReal = 0
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    ⊢ Eq (MeasureTheory.eLpNorm (↑0) p μ).toNNReal 0
  -/
  simp [eLpNorm_congr_ae AEEqFun.coeFn_zero, eLpNorm_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_zero : ‖(0 : Lp E p μ)‖ = 0 :=
  congr_arg ((↑) : ℝ≥0 → ℝ) nnnorm_zero


@[simp]
theorem norm_measure_zero (f : Lp E p (0 : MeasureTheory.Measure α)) : ‖f‖ = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    inst✝ : NormedAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p 0) x
    ⊢ Eq (Norm.norm f) 0
  -/
  simp [norm_def]
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    α : Type u_1
                                                                    E : Type u_2
                                                                    m0 : MeasurableSpace α
                                                                    μ : MeasureTheory.Measure α
                                                                    inst✝ : NormedAddCommGroup E
                                                                    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 0 μ) x
                                                                    ⊢ Eq (Norm.norm f) 0
                                                                  -/
@[simp] theorem norm_exponent_zero (f : Lp E 0 μ) : ‖f‖ = 0 := by simp [norm_def]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem nnnorm_eq_zero_iff {f : Lp E p μ} (hp : 0 < p) : ‖f‖₊ = 0 ↔ f = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hp : LT.lt 0 p
    ⊢ Iff (Eq (NNNorm.nnnorm f) 0) (Eq f 0)
  -/
  refine ⟨fun hf => ?_, fun hf => by simp [hf]⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hp : LT.lt 0 p
    hf : Eq (NNNorm.nnnorm f) 0
    ⊢ Eq f 0
  -/
  rw [nnnorm_def, ENNReal.toNNReal_eq_zero_iff] at hf
  cases hf with
  | inl hf =>
    rw [eLpNorm_eq_zero_iff (Lp.aestronglyMeasurable f) hp.ne.symm] at hf
    exact Subtype.eq (AEEqFun.ext (hf.trans AEEqFun.coeFn_zero.symm))
  | inr hf =>
    exact absurd hf (eLpNorm_ne_top f)


theorem norm_eq_zero_iff {f : Lp E p μ} (hp : 0 < p) : ‖f‖ = 0 ↔ f = 0 :=
  NNReal.coe_eq_zero.trans (nnnorm_eq_zero_iff hp)


theorem eq_zero_iff_ae_eq_zero {f : Lp E p μ} : f = 0 ↔ f =ᵐ[μ] 0 := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff (Eq f 0) ((MeasureTheory.ae μ).EventuallyEq (↑↑f) 0)
  -/
  rw [← (Lp.memℒp f).toLp_eq_toLp_iff zero_memℒp, Memℒp.toLp_zero, toLp_coeFn]
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_neg (f : Lp E p μ) : ‖-f‖₊ = ‖f‖₊ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Eq (NNNorm.nnnorm (Neg.neg f)) (NNNorm.nnnorm f)
  -/
  rw [nnnorm_def, nnnorm_def, eLpNorm_congr_ae (coeFn_neg _), eLpNorm_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_neg (f : Lp E p μ) : ‖-f‖ = ‖f‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) (nnnorm_neg f)


theorem nnnorm_le_mul_nnnorm_of_ae_le_mul {c : ℝ≥0} {f : Lp E p μ} {g : Lp F p μ}
    (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ c * ‖g x‖₊) : ‖f‖₊ ≤ c * ‖g‖₊ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    c : NNReal
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) (HMul.hMul c (NN …
    ⊢ LE.le (NNNorm.nnnorm f) (HMul.hMul c (NNNorm.nnnorm g))
  -/
  simp only [nnnorm_def]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    c : NNReal
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) (HMul.hMul c (NN …
    ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p μ).toNNReal (HMul.hMul c (MeasureTheory …
  -/
  have := eLpNorm_le_nnreal_smul_eLpNorm_of_ae_le_mul h p
  rwa [← ENNReal.toNNReal_le_toNNReal, ENNReal.smul_def, smul_eq_mul, ENNReal.toNNReal_mul,
    ENNReal.toNNReal_coe] at this
    /-
      case ha
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      c : NNReal
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) (HMul.hMul c (NN …
      this : LE.le (MeasureTheory.eLpNorm (↑↑f) p μ) (HSMul.hSMul c (MeasureTheory.e …
      ⊢ Ne (MeasureTheory.eLpNorm (↑↑f) p μ) Top.top
    -/
  · exact (Lp.memℒp _).eLpNorm_ne_top
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      c : NNReal
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      h : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) (HMul.hMul c (NN …
      this : LE.le (MeasureTheory.eLpNorm (↑↑f) p μ) (HSMul.hSMul c (MeasureTheory.e …
      ⊢ Ne (HSMul.hSMul c (MeasureTheory.eLpNorm (↑↑g) p μ)) Top.top
    -/
  · exact ENNReal.mul_ne_top ENNReal.coe_ne_top (Lp.memℒp _).eLpNorm_ne_top
    /-
      🎉 no goals
    -/


theorem norm_le_mul_norm_of_ae_le_mul {c : ℝ} {f : Lp E p μ} {g : Lp F p μ}
    (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ c * ‖g x‖) : ‖f‖ ≤ c * ‖g‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    c : Real
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (HMul.hMul c (Norm.n …
    ⊢ LE.le (Norm.norm f) (HMul.hMul c (Norm.norm g))
  -/
  rcases le_or_lt 0 c with hc | hc
    /-
      case inl
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      c : Real
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (HMul.hMul c (Norm.n …
      hc : LE.le 0 c
      ⊢ LE.le (Norm.norm f) (HMul.hMul c (Norm.norm g))
    -/
  · lift c to ℝ≥0 using hc
    /-
      case inl.intro
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      c : NNReal
      h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (HMul.hMul (↑c) (Nor …
      ⊢ LE.le (Norm.norm f) (HMul.hMul (↑c) (Norm.norm g))
    -/
    exact NNReal.coe_le_coe.mpr (nnnorm_le_mul_nnnorm_of_ae_le_mul h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      c : Real
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (HMul.hMul c (Norm.n …
      hc : LT.lt c 0
      ⊢ LE.le (Norm.norm f) (HMul.hMul c (Norm.norm g))
    -/
  · simp only [norm_def]
    /-
      case inr
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      c : Real
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (HMul.hMul c (Norm.n …
      hc : LT.lt c 0
      ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p μ).toReal (HMul.hMul c (MeasureTheory.e …
    -/
    have := eLpNorm_eq_zero_and_zero_of_ae_le_mul_neg h hc p
    /-
      case inr
      α : Type u_1
      E : Type u_2
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      c : Real
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
      h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (HMul.hMul c (Norm.n …
      hc : LT.lt c 0
      this : And (Eq (MeasureTheory.eLpNorm (↑↑f) p μ) 0) (Eq (MeasureTheory.eLpNorm …
      ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p μ).toReal (HMul.hMul c (MeasureTheory.e …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


theorem norm_le_norm_of_ae_le {f : Lp E p μ} {g : Lp F p μ} (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ ‖g x‖) :
    ‖f‖ ≤ ‖g‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (Norm.norm (↑↑g x))) …
    ⊢ LE.le (Norm.norm f) (Norm.norm g)
  -/
  rw [norm_def, norm_def]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    h : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (Norm.norm (↑↑g x))) …
    ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p μ).toReal (MeasureTheory.eLpNorm (↑↑g)  …
  -/
  exact ENNReal.toReal_mono (eLpNorm_ne_top _) (eLpNorm_mono_ae h)
  /-
    🎉 no goals
  -/


theorem mem_Lp_of_nnnorm_ae_le_mul {c : ℝ≥0} {f : α →ₘ[μ] E} {g : Lp F p μ}
    (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ c * ‖g x‖₊) : f ∈ Lp E p μ :=
  mem_Lp_iff_memℒp.2 <| Memℒp.of_nnnorm_le_mul (Lp.memℒp g) f.aestronglyMeasurable h


theorem mem_Lp_of_ae_le_mul {c : ℝ} {f : α →ₘ[μ] E} {g : Lp F p μ}
    (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ c * ‖g x‖) : f ∈ Lp E p μ :=
  mem_Lp_iff_memℒp.2 <| Memℒp.of_le_mul (Lp.memℒp g) f.aestronglyMeasurable h


theorem mem_Lp_of_nnnorm_ae_le {f : α →ₘ[μ] E} {g : Lp F p μ} (h : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ ‖g x‖₊) :
    f ∈ Lp E p μ :=
  mem_Lp_iff_memℒp.2 <| Memℒp.of_le (Lp.memℒp g) f.aestronglyMeasurable h


theorem mem_Lp_of_ae_le {f : α →ₘ[μ] E} {g : Lp F p μ} (h : ∀ᵐ x ∂μ, ‖f x‖ ≤ ‖g x‖) :
    f ∈ Lp E p μ :=
  mem_Lp_of_nnnorm_ae_le h


theorem mem_Lp_of_ae_nnnorm_bound [IsFiniteMeasure μ] {f : α →ₘ[μ] E} (C : ℝ≥0)
    (hfC : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ C) : f ∈ Lp E p μ :=
  mem_Lp_iff_memℒp.2 <| Memℒp.of_bound f.aestronglyMeasurable _ hfC


theorem mem_Lp_of_ae_bound [IsFiniteMeasure μ] {f : α →ₘ[μ] E} (C : ℝ) (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) :
    f ∈ Lp E p μ :=
  mem_Lp_iff_memℒp.2 <| Memℒp.of_bound f.aestronglyMeasurable _ hfC


theorem nnnorm_le_of_ae_bound [IsFiniteMeasure μ] {f : Lp E p μ} {C : ℝ≥0}
    (hfC : ∀ᵐ x ∂μ, ‖f x‖₊ ≤ C) : ‖f‖₊ ≤ measureUnivNNReal μ ^ p.toReal⁻¹ * C := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) C) (MeasureThe …
    ⊢ LE.le (NNNorm.nnnorm f) (HMul.hMul (HPow.hPow (MeasureTheory.measureUnivNNRe …
  -/
  by_cases hμ : μ = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      C : NNReal
      hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) C) (MeasureThe …
      hμ : Eq μ 0
      ⊢ LE.le (NNNorm.nnnorm f) (HMul.hMul (HPow.hPow (MeasureTheory.measureUnivNNRe …
    -/
  · by_cases hp : p.toReal⁻¹ = 0
      /-
        case pos
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup E
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        C : NNReal
        hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) C) (MeasureThe …
        hμ : Eq μ 0
        hp : Eq (Inv.inv p.toReal) 0
        ⊢ LE.le (NNNorm.nnnorm f) (HMul.hMul (HPow.hPow (MeasureTheory.measureUnivNNRe …
      -/
    · simp [hp, hμ, nnnorm_def]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup E
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        C : NNReal
        hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) C) (MeasureThe …
        hμ : Eq μ 0
        hp : Not (Eq (Inv.inv p.toReal) 0)
        ⊢ LE.le (NNNorm.nnnorm f) (HMul.hMul (HPow.hPow (MeasureTheory.measureUnivNNRe …
      -/
    · simp [hμ, nnnorm_def, Real.zero_rpow hp]
      /-
        🎉 no goals
      -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) C) (MeasureThe …
    hμ : Not (Eq μ 0)
    ⊢ LE.le (NNNorm.nnnorm f) (HMul.hMul (HPow.hPow (MeasureTheory.measureUnivNNRe …
  -/
  rw [← ENNReal.coe_le_coe, nnnorm_def, ENNReal.coe_toNNReal (eLpNorm_ne_top _)]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑f x)) C) (MeasureThe …
    hμ : Not (Eq μ 0)
    ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p μ) ↑(HMul.hMul (HPow.hPow (MeasureTheor …
  -/
  refine (eLpNorm_le_of_ae_nnnorm_bound hfC).trans_eq ?_
  rw [← coe_measureUnivNNReal μ, ← ENNReal.coe_rpow_of_ne_zero (measureUnivNNReal_pos hμ).ne',
    ENNReal.coe_mul, mul_comm, ENNReal.smul_def, smul_eq_mul]


theorem norm_le_of_ae_bound [IsFiniteMeasure μ] {f : Lp E p μ} {C : ℝ} (hC : 0 ≤ C)
    (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) : ‖f‖ ≤ measureUnivNNReal μ ^ p.toReal⁻¹ * C := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    C : Real
    hC : LE.le 0 C
    hfC : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) C) (MeasureTheory. …
    ⊢ LE.le (Norm.norm f) (HMul.hMul (HPow.hPow (↑(MeasureTheory.measureUnivNNReal …
  -/
  lift C to ℝ≥0 using hC
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) ↑C) (MeasureTheory …
    ⊢ LE.le (Norm.norm f) (HMul.hMul (HPow.hPow (↑(MeasureTheory.measureUnivNNReal …
  -/
  have := nnnorm_le_of_ae_bound hfC
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    C : NNReal
    hfC : Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) ↑C) (MeasureTheory …
    this : LE.le (NNNorm.nnnorm f) (HMul.hMul (HPow.hPow (MeasureTheory.measureUni …
    ⊢ LE.le (Norm.norm f) (HMul.hMul (HPow.hPow (↑(MeasureTheory.measureUnivNNReal …
  -/
  rwa [← NNReal.coe_le_coe, NNReal.coe_mul, NNReal.coe_rpow] at this
  /-
    🎉 no goals
  -/


instance instNormedAddCommGroup [hp : Fact (1 ≤ p)] : NormedAddCommGroup (Lp E p μ) :=
  { AddGroupNorm.toNormedAddCommGroup
      { toFun := (norm : Lp E p μ → ℝ)
        map_zero' := norm_zero
                   /-
                     α : Type u_1
                     E : Type u_2
                     F : Type u_3
                     G : Type u_4
                     m m0 : MeasurableSpace α
                     p : ENNReal
                     q : Real
                     μ ν : MeasureTheory.Measure α
                     inst✝² : NormedAddCommGroup E
                     inst✝¹ : NormedAddCommGroup F
                     inst✝ : NormedAddCommGroup G
                     hp : Fact (LE.le 1 p)
                     ⊢ ∀ (r : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x), Eq (Norm …
                   -/
        neg' := by simp
          /-
            α : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            m m0 : MeasurableSpace α
            p : ENNReal
            q : Real
            μ ν : MeasureTheory.Measure α
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedAddCommGroup F
            inst✝ : NormedAddCommGroup G
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
            ⊢ LE.le (Norm.norm (HAdd.hAdd f g)) (HAdd.hAdd (Norm.norm f) (Norm.norm g))
          -/
                   /-
                     🎉 no goals
                   -/
          /-
            α : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            m m0 : MeasurableSpace α
            p : ENNReal
            q : Real
            μ ν : MeasureTheory.Measure α
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedAddCommGroup F
            inst✝ : NormedAddCommGroup G
            hp : Fact (LE.le 1 p)
            f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
            ⊢ LE.le (↑(NNNorm.nnnorm (HAdd.hAdd f g))) (HAdd.hAdd ↑(NNNorm.nnnorm f) ↑(NNN …
          -/
        add_le' := fun f g => by
          suffices (‖f + g‖₊ : ℝ≥0∞) ≤ ‖f‖₊ + ‖g‖₊ from mod_cast this
          simp only [Lp.nnnorm_coe_ennreal]
          exact (eLpNorm_congr_ae (AEEqFun.coeFn_add _ _)).trans_le
            (eLpNorm_add_le (Lp.aestronglyMeasurable _) (Lp.aestronglyMeasurable _) hp.out)
        eq_zero_of_map_eq_zero' := fun _ =>
          (norm_eq_zero_iff <| zero_lt_one.trans_le hp.1).1 } with
    edist := edist
    edist_dist := Lp.edist_dist }

-- check no diamond is created

theorem const_smul_mem_Lp (c : 𝕜) (f : Lp E p μ) : c • (f : α →ₘ[μ] E) ∈ Lp E p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_5
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Membership.mem (MeasureTheory.Lp E p μ) (HSMul.hSMul c ↑f)
  -/
  rw [mem_Lp_iff_eLpNorm_lt_top, eLpNorm_congr_ae (AEEqFun.coeFn_smul _ _)]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_5
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSMul.hSMul c ↑↑f) p μ) Top.top
  -/
  refine eLpNorm_const_smul_le.trans_lt ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_5
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ LT.lt (HSMul.hSMul (NNNorm.nnnorm c) (MeasureTheory.eLpNorm (↑↑f) p μ)) Top. …
  -/
  rw [ENNReal.smul_def, smul_eq_mul, ENNReal.mul_lt_top_iff]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_5
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Or (And (LT.lt (↑(NNNorm.nnnorm c)) Top.top) (LT.lt (MeasureTheory.eLpNorm ( …
  -/
  exact Or.inl ⟨ENNReal.coe_lt_top, f.prop⟩
  /-
    🎉 no goals
  -/


/-- The `𝕜`-submodule of elements of `α →ₘ[μ] E` whose `Lp` norm is finite.  This is `Lp E p μ`,
with extra structure. -/
def LpSubmodule : Submodule 𝕜 (α →ₘ[μ] E) :=
                                                /-
                                                  α : Type u_1
                                                  E : Type u_2
                                                  F : Type u_3
                                                  G : Type u_4
                                                  m m0 : MeasurableSpace α
                                                  p : ENNReal
                                                  q : Real
                                                  μ ν : MeasureTheory.Measure α
                                                  inst✝⁸ : NormedAddCommGroup E
                                                  inst✝⁷ : NormedAddCommGroup F
                                                  inst✝⁶ : NormedAddCommGroup G
                                                  𝕜 : Type u_5
                                                  𝕜' : Type u_6
                                                  inst✝⁵ : NormedRing 𝕜
                                                  inst✝⁴ : NormedRing 𝕜'
                                                  inst✝³ : Module 𝕜 E
                                                  inst✝² : Module 𝕜' E
                                                  inst✝¹ : BoundedSMul 𝕜 E
                                                  inst✝ : BoundedSMul 𝕜' E
                                                  c : 𝕜
                                                  f : MeasureTheory.AEEqFun α E μ
                                                  hf : Membership.mem __src✝.carrier f
                                                  ⊢ Membership.mem __src✝.carrier (HSMul.hSMul c f)
                                                -/
  { Lp E p μ with smul_mem' := fun c f hf => by simpa using const_smul_mem_Lp c ⟨f, hf⟩ }
                                                /-
                                                  🎉 no goals
                                                -/


theorem coe_LpSubmodule : (LpSubmodule E p μ 𝕜).toAddSubgroup = Lp E p μ :=
  rfl


instance instModule : Module 𝕜 (Lp E p μ) :=
  { (LpSubmodule E p μ 𝕜).module with }


theorem coeFn_smul (c : 𝕜) (f : Lp E p μ) : ⇑(c • f) =ᵐ[μ] c • ⇑f :=
  AEEqFun.coeFn_smul _ _


instance instIsCentralScalar [Module 𝕜ᵐᵒᵖ E] [BoundedSMul 𝕜ᵐᵒᵖ E] [IsCentralScalar 𝕜 E] :
    IsCentralScalar 𝕜 (Lp E p μ) where
  op_smul_eq_smul k f := Subtype.ext <| op_smul_eq_smul k (f : α →ₘ[μ] E)


instance instSMulCommClass [SMulCommClass 𝕜 𝕜' E] : SMulCommClass 𝕜 𝕜' (Lp E p μ) where
  smul_comm k k' f := Subtype.ext <| smul_comm k k' (f : α →ₘ[μ] E)


instance instIsScalarTower [SMul 𝕜 𝕜'] [IsScalarTower 𝕜 𝕜' E] : IsScalarTower 𝕜 𝕜' (Lp E p μ) where
  smul_assoc k k' f := Subtype.ext <| smul_assoc k k' (f : α →ₘ[μ] E)


instance instBoundedSMul [Fact (1 ≤ p)] : BoundedSMul 𝕜 (Lp E p μ) :=
  -- TODO: add `BoundedSMul.of_nnnorm_smul_le`
  BoundedSMul.of_norm_smul_le fun r f => by
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedAddCommGroup G
      𝕜 : Type u_5
      𝕜' : Type u_6
      inst✝⁶ : NormedRing 𝕜
      inst✝⁵ : NormedRing 𝕜'
      inst✝⁴ : Module 𝕜 E
      inst✝³ : Module 𝕜' E
      inst✝² : BoundedSMul 𝕜 E
      inst✝¹ : BoundedSMul 𝕜' E
      inst✝ : Fact (LE.le 1 p)
      r : 𝕜
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ LE.le (Norm.norm (HSMul.hSMul r f)) (HMul.hMul (Norm.norm r) (Norm.norm f))
    -/
    suffices (‖r • f‖₊ : ℝ≥0∞) ≤ ‖r‖₊ * ‖f‖₊ from mod_cast this
    rw [nnnorm_def, nnnorm_def, ENNReal.coe_toNNReal (Lp.eLpNorm_ne_top _),
      eLpNorm_congr_ae (coeFn_smul _ _), ENNReal.coe_toNNReal (Lp.eLpNorm_ne_top _)]
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedAddCommGroup G
      𝕜 : Type u_5
      𝕜' : Type u_6
      inst✝⁶ : NormedRing 𝕜
      inst✝⁵ : NormedRing 𝕜'
      inst✝⁴ : Module 𝕜 E
      inst✝³ : Module 𝕜' E
      inst✝² : BoundedSMul 𝕜 E
      inst✝¹ : BoundedSMul 𝕜' E
      inst✝ : Fact (LE.le 1 p)
      r : 𝕜
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ LE.le (MeasureTheory.eLpNorm (HSMul.hSMul r ↑↑f) p μ) (HMul.hMul (↑(NNNorm.n …
    -/
    exact eLpNorm_const_smul_le
    /-
      🎉 no goals
    -/


instance instNormedSpace [Fact (1 ≤ p)] : NormedSpace 𝕜 (Lp E p μ) where
  norm_smul_le _ _ := norm_smul_le _ _


theorem toLp_const_smul {f : α → E} (c : 𝕜) (hf : Memℒp f p μ) :
    (hf.const_smul c).toLp (c • f) = c • hf.toLp f :=
  rfl


theorem eLpNormEssSup_indicator_le (s : Set α) (f : α → G) :
    eLpNormEssSup (s.indicator f) μ ≤ eLpNormEssSup f μ := by
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    f : α → G
    ⊢ LE.le (MeasureTheory.eLpNormEssSup (s.indicator f) μ) (MeasureTheory.eLpNorm …
  -/
  refine essSup_mono_ae (Eventually.of_forall fun x => ?_)
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    f : α → G
    x : α
    ⊢ LE.le ((fun x => ENorm.enorm (s.indicator f x)) x) ((fun x => ENorm.enorm (f …
  -/
  simp_rw [enorm_eq_nnnorm, ENNReal.coe_le_coe, nnnorm_indicator_eq_indicator_nnnorm]
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    f : α → G
    x : α
    ⊢ LE.le (s.indicator (fun a => NNNorm.nnnorm (f a)) x) (NNNorm.nnnorm (f x))
  -/
  exact Set.indicator_le_self s _ x
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_indicator_le := eLpNormEssSup_indicator_le


theorem eLpNormEssSup_indicator_const_le (s : Set α) (c : G) :
    eLpNormEssSup (s.indicator fun _ : α => c) μ ≤ ‖c‖₊ := by
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    ⊢ LE.le (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nnno …
  -/
  by_cases hμ0 : μ = 0
    /-
      case pos
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      hμ0 : Eq μ 0
      ⊢ LE.le (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nnno …
    -/
  · rw [hμ0, eLpNormEssSup_measure_zero]
    /-
      case pos
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      hμ0 : Eq μ 0
      ⊢ LE.le 0 ↑(NNNorm.nnnorm c)
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      hμ0 : Not (Eq μ 0)
      ⊢ LE.le (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nnno …
    -/
  · exact (eLpNormEssSup_indicator_le s fun _ => c).trans (eLpNormEssSup_const c hμ0).le
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_indicator_const_le := eLpNormEssSup_indicator_const_le


theorem eLpNormEssSup_indicator_const_eq (s : Set α) (c : G) (hμs : μ s ≠ 0) :
    eLpNormEssSup (s.indicator fun _ : α => c) μ = ‖c‖₊ := by
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hμs : Ne (μ s) 0
    ⊢ Eq (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nnnorm c)
  -/
  refine le_antisymm (eLpNormEssSup_indicator_const_le s c) ?_
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hμs : Ne (μ s) 0
    ⊢ LE.le (↑(NNNorm.nnnorm c)) (MeasureTheory.eLpNormEssSup (s.indicator fun x = …
  -/
  by_contra! h
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hμs : Ne (μ s) 0
    h : LT.lt (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nn …
    ⊢ False
  -/
  have h' := ae_iff.mp (ae_lt_of_essSup_lt h)
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hμs : Ne (μ s) 0
    h : LT.lt (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nn …
    h' : Eq (μ (setOf fun a => Not (LT.lt (ENorm.enorm (s.indicator (fun x => c) a …
    ⊢ False
  -/
  push_neg at h'
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hμs : Ne (μ s) 0
    h : LT.lt (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nn …
    h' : Eq (μ (setOf fun a => LE.le (↑(NNNorm.nnnorm c)) (ENorm.enorm (s.indicato …
    ⊢ False
  -/
  refine hμs (measure_mono_null (fun x hx_mem => ?_) h')
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hμs : Ne (μ s) 0
    h : LT.lt (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nn …
    h' : Eq (μ (setOf fun a => LE.le (↑(NNNorm.nnnorm c)) (ENorm.enorm (s.indicato …
    x : α
    hx_mem : Membership.mem s x
    ⊢ Membership.mem (setOf fun a => LE.le (↑(NNNorm.nnnorm c)) (ENorm.enorm (s.in …
  -/
  rw [Set.mem_setOf_eq, Set.indicator_of_mem hx_mem, enorm_eq_nnnorm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snormEssSup_indicator_const_eq := eLpNormEssSup_indicator_const_eq


theorem eLpNorm_indicator_le (f : α → E) : eLpNorm (s.indicator f) p μ ≤ eLpNorm f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    f : α → E
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p …
  -/
  refine eLpNorm_mono_ae (Eventually.of_forall fun x => ?_)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    f : α → E
    x : α
    ⊢ LE.le (Norm.norm (s.indicator f x)) (Norm.norm (f x))
  -/
  suffices ‖s.indicator f x‖₊ ≤ ‖f x‖₊ by exact NNReal.coe_mono this
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    f : α → E
    x : α
    ⊢ LE.le (NNNorm.nnnorm (s.indicator f x)) (NNNorm.nnnorm (f x))
  -/
  rw [nnnorm_indicator_eq_indicator_nnnorm]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    f : α → E
    x : α
    ⊢ LE.le (s.indicator (fun a => NNNorm.nnnorm (f a)) x) (NNNorm.nnnorm (f x))
  -/
  exact s.indicator_le_self _ x
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_le := eLpNorm_indicator_le


lemma eLpNorm_indicator_const₀ {c : G} (hs : NullMeasurableSet s μ) (hp : p ≠ 0) (hp_top : p ≠ ∞) :
    eLpNorm (s.indicator fun _ => c) p μ = ‖c‖₊ * μ s ^ (1 / p.toReal) :=
  have hp_pos : 0 < p.toReal := ENNReal.toReal_pos hp hp_top
  calc
    eLpNorm (s.indicator fun _ => c) p μ
      = (∫⁻ x, ((‖(s.indicator fun _ ↦ c) x‖₊ : ℝ≥0∞) ^ p.toReal) ∂μ) ^ (1 / p.toReal) :=
          eLpNorm_eq_lintegral_rpow_nnnorm hp hp_top
    _ = (∫⁻ x, (s.indicator fun _ ↦ (‖c‖₊ : ℝ≥0∞) ^ p.toReal) x ∂μ) ^ (1 / p.toReal) := by
      /-
        α : Type u_1
        G : Type u_4
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup G
        s : Set α
        c : G
        hs : MeasureTheory.NullMeasurableSet s μ
        hp : Ne p 0
        hp_top : Ne p Top.top
        hp_pos : LT.lt 0 p.toReal
        ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm …
      -/
      congr 2
      /-
        case e_a.e_f
        α : Type u_1
        G : Type u_4
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup G
        s : Set α
        c : G
        hs : MeasureTheory.NullMeasurableSet s μ
        hp : Ne p 0
        hp_top : Ne p Top.top
        hp_pos : LT.lt 0 p.toReal
        ⊢ Eq (fun x => HPow.hPow (↑(NNNorm.nnnorm (s.indicator (fun x => c) x))) p.toR …
      -/
      refine (Set.comp_indicator_const c (fun x : G ↦ (‖x‖₊ : ℝ≥0∞) ^ p.toReal) ?_)
      /-
        case e_a.e_f
        α : Type u_1
        G : Type u_4
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup G
        s : Set α
        c : G
        hs : MeasureTheory.NullMeasurableSet s μ
        hp : Ne p 0
        hp_top : Ne p Top.top
        hp_pos : LT.lt 0 p.toReal
        ⊢ Eq ((fun x => HPow.hPow (↑(NNNorm.nnnorm x)) p.toReal) 0) 0
      -/
      simp [hp_pos]
      /-
        🎉 no goals
      -/
    _ = ‖c‖₊ * μ s ^ (1 / p.toReal) := by
      rw [lintegral_indicator_const₀ hs, ENNReal.mul_rpow_of_nonneg, ← ENNReal.rpow_mul,
        mul_one_div_cancel hp_pos.ne', ENNReal.rpow_one]
      /-
        case hz
        α : Type u_1
        G : Type u_4
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup G
        s : Set α
        c : G
        hs : MeasureTheory.NullMeasurableSet s μ
        hp : Ne p 0
        hp_top : Ne p Top.top
        hp_pos : LT.lt 0 p.toReal
        ⊢ LE.le 0 (HDiv.hDiv 1 p.toReal)
      -/
      positivity
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_const₀ := eLpNorm_indicator_const₀


theorem eLpNorm_indicator_const {c : G} (hs : MeasurableSet s) (hp : p ≠ 0) (hp_top : p ≠ ∞) :
    eLpNorm (s.indicator fun _ => c) p μ = ‖c‖₊ * μ s ^ (1 / p.toReal) :=
  eLpNorm_indicator_const₀ hs.nullMeasurableSet hp hp_top


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_const := eLpNorm_indicator_const


theorem eLpNorm_indicator_const' {c : G} (hs : MeasurableSet s) (hμs : μ s ≠ 0) (hp : p ≠ 0) :
    eLpNorm (s.indicator fun _ => c) p μ = ‖c‖₊ * μ s ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    hs : MeasurableSet s
    hμs : Ne (μ s) 0
    hp : Ne p 0
    ⊢ Eq (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (↑(NNNorm …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      hs : MeasurableSet s
      hμs : Ne (μ s) 0
      hp : Ne p 0
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (↑(NNNorm …
    -/
  · simp [hp_top, eLpNormEssSup_indicator_const_eq s c hμs]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      hs : MeasurableSet s
      hμs : Ne (μ s) 0
      hp : Ne p 0
      hp_top : Not (Eq p Top.top)
      ⊢ Eq (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (↑(NNNorm …
    -/
  · exact eLpNorm_indicator_const hs hp hp_top
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_const' := eLpNorm_indicator_const'


theorem eLpNorm_indicator_const_le (c : G) (p : ℝ≥0∞) :
    eLpNorm (s.indicator fun _ => c) p μ ≤ ‖c‖₊ * μ s ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    p : ENNReal
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (↑(NNN …
  -/
  rcases eq_or_ne p 0 with (rfl | hp)
    /-
      case inl
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) 0 μ) (HMul.hMul (↑(NNN …
    -/
  · simp only [eLpNorm_exponent_zero, zero_le']
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    p : ENNReal
    hp : Ne p 0
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (↑(NNN …
  -/
  rcases eq_or_ne p ∞ with (rfl | h'p)
  · simp only [eLpNorm_exponent_top, ENNReal.top_toReal, _root_.div_zero, ENNReal.rpow_zero,
      mul_one]
    /-
      case inr.inl
      α : Type u_1
      G : Type u_4
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup G
      s : Set α
      c : G
      hp : Ne Top.top 0
      ⊢ LE.le (MeasureTheory.eLpNormEssSup (s.indicator fun x => c) μ) ↑(NNNorm.nnno …
    -/
    exact eLpNormEssSup_indicator_const_le _ _
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup G
    s : Set α
    c : G
    p : ENNReal
    hp : Ne p 0
    h'p : Ne p Top.top
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (HMul.hMul (↑(NNN …
  -/
  let t := toMeasurable μ s
  calc
    eLpNorm (s.indicator fun _ => c) p μ ≤ eLpNorm (t.indicator fun _ => c) p μ :=
      eLpNorm_mono (norm_indicator_le_of_subset (subset_toMeasurable _ _) _)
    _ = ‖c‖₊ * μ t ^ (1 / p.toReal) :=
      (eLpNorm_indicator_const (measurableSet_toMeasurable _ _) hp h'p)
    _ = ‖c‖₊ * μ s ^ (1 / p.toReal) := by rw [measure_toMeasurable]


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_const_le := eLpNorm_indicator_const_le


theorem Memℒp.indicator (hs : MeasurableSet s) (hf : Memℒp f p μ) : Memℒp (s.indicator f) p μ :=
  ⟨hf.aestronglyMeasurable.indicator hs, lt_of_le_of_lt (eLpNorm_indicator_le f) hf.eLpNorm_lt_top⟩


theorem eLpNormEssSup_indicator_eq_eLpNormEssSup_restrict {f : α → F} (hs : MeasurableSet s) :
    eLpNormEssSup (s.indicator f) μ = eLpNormEssSup f (μ.restrict s) := by
  simp_rw [eLpNormEssSup_eq_essSup_nnnorm, nnnorm_indicator_eq_indicator_nnnorm,
    ENNReal.coe_indicator, ENNReal.essSup_indicator_eq_essSup_restrict hs]


@[deprecated (since := "2024-07-27")]
alias snormEssSup_indicator_eq_snormEssSup_restrict :=
  eLpNormEssSup_indicator_eq_eLpNormEssSup_restrict


theorem eLpNorm_indicator_eq_eLpNorm_restrict {f : α → F} (hs : MeasurableSet s) :
    eLpNorm (s.indicator f) p μ = eLpNorm f p (μ.restrict s) := by
  /-
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      α : Type u_1
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hs : MeasurableSet s
      hp_zero : Eq p 0
      ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
    -/
  · simp only [hp_zero, eLpNorm_exponent_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    hp_zero : Not (Eq p 0)
    ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hs : MeasurableSet s
      hp_zero : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
    -/
  · simp_rw [hp_top, eLpNorm_exponent_top]
    /-
      case pos
      α : Type u_1
      F : Type u_3
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      s : Set α
      f : α → F
      hs : MeasurableSet s
      hp_zero : Not (Eq p 0)
      hp_top : Eq p Top.top
      ⊢ Eq (MeasureTheory.eLpNormEssSup (s.indicator f) μ) (MeasureTheory.eLpNormEss …
    -/
    exact eLpNormEssSup_indicator_eq_eLpNormEssSup_restrict hs
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.eLpNorm (s.indicator f) p μ) (MeasureTheory.eLpNorm f p (μ …
  -/
  simp_rw [eLpNorm_eq_lintegral_rpow_nnnorm hp_zero hp_top]
  suffices (∫⁻ x, (‖s.indicator f x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ) =
      ∫⁻ x in s, (‖f x‖₊ : ℝ≥0∞) ^ p.toReal ∂μ by rw [this]
  /-
    case neg
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm (s.indicat …
  -/
  rw [← lintegral_indicator hs]
  /-
    case neg
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (MeasureTheory.lintegral μ fun x => HPow.hPow (↑(NNNorm.nnnorm (s.indicat …
  -/
  congr
  /-
    case neg.e_f
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    ⊢ Eq (fun x => HPow.hPow (↑(NNNorm.nnnorm (s.indicator f x))) p.toReal) fun a  …
  -/
  simp_rw [nnnorm_indicator_eq_indicator_nnnorm, ENNReal.coe_indicator]
  have h_zero : (fun x => x ^ p.toReal) (0 : ℝ≥0∞) = 0 := by
    simp [ENNReal.toReal_pos hp_zero hp_top]
  -- Porting note: The implicit argument should be specified because the elaborator can't deal with
  --               `∘` well.
  /-
    case neg.e_f
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    s : Set α
    f : α → F
    hs : MeasurableSet s
    hp_zero : Not (Eq p 0)
    hp_top : Not (Eq p Top.top)
    h_zero : Eq ((fun x => HPow.hPow x p.toReal) 0) 0
    ⊢ Eq (fun x => HPow.hPow (s.indicator (fun x => ↑(NNNorm.nnnorm (f x))) x) p.t …
  -/
  exact (Set.indicator_comp_of_zero (g := fun x : ℝ≥0∞ => x ^ p.toReal) h_zero).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_indicator_eq_snorm_restrict := eLpNorm_indicator_eq_eLpNorm_restrict


theorem memℒp_indicator_iff_restrict (hs : MeasurableSet s) :
    Memℒp (s.indicator f) p μ ↔ Memℒp f p (μ.restrict s) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    s : Set α
    hs : MeasurableSet s
    ⊢ Iff (MeasureTheory.Memℒp (s.indicator f) p μ) (MeasureTheory.Memℒp f p (μ.re …
  -/
  simp [Memℒp, aestronglyMeasurable_indicator_iff hs, eLpNorm_indicator_eq_eLpNorm_restrict hs]
  /-
    🎉 no goals
  -/


/-- If a function is supported on a finite-measure set and belongs to `ℒ^p`, then it belongs to
`ℒ^q` for any `q ≤ p`. -/
theorem Memℒp.memℒp_of_exponent_le_of_measure_support_ne_top
    {p q : ℝ≥0∞} {f : α → E} (hfq : Memℒp f q μ) {s : Set α} (hf : ∀ x, x ∉ s → f x = 0)
    (hs : μ s ≠ ∞) (hpq : p ≤ q) : Memℒp f p μ := by
  have : (toMeasurable μ s).indicator f = f := by
    apply Set.indicator_eq_self.2
    apply Function.support_subset_iff'.2 (fun x hx ↦ hf x ?_)
    contrapose! hx
    exact subset_toMeasurable μ s hx
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p q : ENNReal
    f : α → E
    hfq : MeasureTheory.Memℒp f q μ
    s : Set α
    hf : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    hs : Ne (μ s) Top.top
    hpq : LE.le p q
    this : Eq ((MeasureTheory.toMeasurable μ s).indicator f) f
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  rw [← this, memℒp_indicator_iff_restrict (measurableSet_toMeasurable μ s)] at hfq ⊢
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p q : ENNReal
    f : α → E
    s : Set α
    hfq : MeasureTheory.Memℒp f q (μ.restrict (MeasureTheory.toMeasurable μ s))
    hf : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    hs : Ne (μ s) Top.top
    hpq : LE.le p q
    this : Eq ((MeasureTheory.toMeasurable μ s).indicator f) f
    ⊢ MeasureTheory.Memℒp f p (μ.restrict (MeasureTheory.toMeasurable μ s))
  -/
  have : Fact (μ (toMeasurable μ s) < ∞) := ⟨by simpa [lt_top_iff_ne_top] using hs⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    p q : ENNReal
    f : α → E
    s : Set α
    hfq : MeasureTheory.Memℒp f q (μ.restrict (MeasureTheory.toMeasurable μ s))
    hf : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    hs : Ne (μ s) Top.top
    hpq : LE.le p q
    this✝ : Eq ((MeasureTheory.toMeasurable μ s).indicator f) f
    this : Fact (LT.lt (μ (MeasureTheory.toMeasurable μ s)) Top.top)
    ⊢ MeasureTheory.Memℒp f p (μ.restrict (MeasureTheory.toMeasurable μ s))
  -/
  exact memℒp_of_exponent_le hfq hpq
  /-
    🎉 no goals
  -/


theorem memℒp_indicator_const (p : ℝ≥0∞) (hs : MeasurableSet s) (c : E) (hμsc : c = 0 ∨ μ s ≠ ∞) :
    Memℒp (s.indicator fun _ => c) p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    p : ENNReal
    hs : MeasurableSet s
    c : E
    hμsc : Or (Eq c 0) (Ne (μ s) Top.top)
    ⊢ MeasureTheory.Memℒp (s.indicator fun x => c) p μ
  -/
  rw [memℒp_indicator_iff_restrict hs]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    p : ENNReal
    hs : MeasurableSet s
    c : E
    hμsc : Or (Eq c 0) (Ne (μ s) Top.top)
    ⊢ MeasureTheory.Memℒp (fun x => c) p (μ.restrict s)
  -/
  rcases hμsc with rfl | hμ
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      p : ENNReal
      hs : MeasurableSet s
      ⊢ MeasureTheory.Memℒp (fun x => 0) p (μ.restrict s)
    -/
  · exact zero_memℒp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      p : ENNReal
      hs : MeasurableSet s
      c : E
      hμ : Ne (μ s) Top.top
      ⊢ MeasureTheory.Memℒp (fun x => c) p (μ.restrict s)
    -/
  · have := Fact.mk hμ.lt_top
    /-
      case inr
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      p : ENNReal
      hs : MeasurableSet s
      c : E
      hμ : Ne (μ s) Top.top
      this : Fact (LT.lt (μ s) Top.top)
      ⊢ MeasureTheory.Memℒp (fun x => c) p (μ.restrict s)
    -/
    apply memℒp_const
    /-
      🎉 no goals
    -/


/-- The `ℒ^p` norm of the indicator of a set is uniformly small if the set itself has small measure,
for any `p < ∞`. Given here as an existential `∀ ε > 0, ∃ η > 0, ...` to avoid later
management of `ℝ≥0∞`-arithmetic. -/
theorem exists_eLpNorm_indicator_le (hp : p ≠ ∞) (c : E) {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ η : ℝ≥0, 0 < η ∧ ∀ s : Set α, μ s ≤ η → eLpNorm (s.indicator fun _ => c) p μ ≤ ε := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (s : Set α), LE.le (μ s) ↑η → LE.le (Meas …
  -/
  rcases eq_or_ne p 0 with (rfl | h'p)
    /-
      case inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      c : E
      ε : ENNReal
      hε : Ne ε 0
      hp : Ne 0 Top.top
      ⊢ Exists fun η => And (LT.lt 0 η) (∀ (s : Set α), LE.le (μ s) ↑η → LE.le (Meas …
    -/
  · exact ⟨1, zero_lt_one, fun s _ => by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    h'p : Ne p 0
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (s : Set α), LE.le (μ s) ↑η → LE.le (Meas …
  -/
  have hp₀ : 0 < p := bot_lt_iff_ne_bot.2 h'p
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    h'p : Ne p 0
    hp₀ : LT.lt 0 p
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (s : Set α), LE.le (μ s) ↑η → LE.le (Meas …
  -/
  have hp₀' : 0 ≤ 1 / p.toReal := div_nonneg zero_le_one ENNReal.toReal_nonneg
  /-
    case inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    h'p : Ne p 0
    hp₀ : LT.lt 0 p
    hp₀' : LE.le 0 (HDiv.hDiv 1 p.toReal)
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (s : Set α), LE.le (μ s) ↑η → LE.le (Meas …
  -/
  have hp₀'' : 0 < p.toReal := ENNReal.toReal_pos hp₀.ne' hp
  obtain ⟨η, hη_pos, hη_le⟩ :
      ∃ η : ℝ≥0, 0 < η ∧ (‖c‖₊ : ℝ≥0∞) * (η : ℝ≥0∞) ^ (1 / p.toReal) ≤ ε := by
    have :
      Filter.Tendsto (fun x : ℝ≥0 => ((‖c‖₊ * x ^ (1 / p.toReal) : ℝ≥0) : ℝ≥0∞)) (𝓝 0)
        (𝓝 (0 : ℝ≥0)) := by
      rw [ENNReal.tendsto_coe]
      convert (NNReal.continuousAt_rpow_const (Or.inr hp₀')).tendsto.const_mul _
      simp [hp₀''.ne']
    have hε' : 0 < ε := hε.bot_lt
    obtain ⟨δ, hδ, hδε'⟩ := NNReal.nhds_zero_basis.eventually_iff.mp (this.eventually_le_const hε')
    obtain ⟨η, hη, hηδ⟩ := exists_between hδ
    refine ⟨η, hη, ?_⟩
    rw [← ENNReal.coe_rpow_of_nonneg _ hp₀', ← ENNReal.coe_mul]
    exact hδε' hηδ
  /-
    case inr.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    h'p : Ne p 0
    hp₀ : LT.lt 0 p
    hp₀' : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hp₀'' : LT.lt 0 p.toReal
    η : NNReal
    hη_pos : LT.lt 0 η
    hη_le : LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (↑η) (HDiv.hDiv 1 p.t …
    ⊢ Exists fun η => And (LT.lt 0 η) (∀ (s : Set α), LE.le (μ s) ↑η → LE.le (Meas …
  -/
  refine ⟨η, hη_pos, fun s hs => ?_⟩
  /-
    case inr.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    h'p : Ne p 0
    hp₀ : LT.lt 0 p
    hp₀' : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hp₀'' : LT.lt 0 p.toReal
    η : NNReal
    hη_pos : LT.lt 0 η
    hη_le : LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (↑η) (HDiv.hDiv 1 p.t …
    s : Set α
    hs : LE.le (μ s) ↑η
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) ε
  -/
  refine (eLpNorm_indicator_const_le _ _).trans (le_trans ?_ hη_le)
  /-
    case inr.intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Ne p Top.top
    c : E
    ε : ENNReal
    hε : Ne ε 0
    h'p : Ne p 0
    hp₀ : LT.lt 0 p
    hp₀' : LE.le 0 (HDiv.hDiv 1 p.toReal)
    hp₀'' : LT.lt 0 p.toReal
    η : NNReal
    hη_pos : LT.lt 0 η
    hη_le : LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (↑η) (HDiv.hDiv 1 p.t …
    s : Set α
    hs : LE.le (μ s) ↑η
    ⊢ LE.le (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ s) (HDiv.hDiv 1 p.toReal …
  -/
  exact mul_le_mul_left' (ENNReal.rpow_le_rpow hs hp₀') _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias exists_snorm_indicator_le := exists_eLpNorm_indicator_le


protected lemma Memℒp.piecewise [DecidablePred (· ∈ s)] {g}
    (hs : MeasurableSet s) (hf : Memℒp f p (μ.restrict s)) (hg : Memℒp g p (μ.restrict sᶜ)) :
    Memℒp (s.piecewise f g) p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    g : α → E
    hs : MeasurableSet s
    hf : MeasureTheory.Memℒp f p (μ.restrict s)
    hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
    ⊢ MeasureTheory.Memℒp (s.piecewise f g) p μ
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f p (μ.restrict s)
      hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
      hp_zero : Eq p 0
      ⊢ MeasureTheory.Memℒp (s.piecewise f g) p μ
    -/
  · simp only [hp_zero, memℒp_zero_iff_aestronglyMeasurable]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f p (μ.restrict s)
      hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
      hp_zero : Eq p 0
      ⊢ MeasureTheory.AEStronglyMeasurable (s.piecewise f g) μ
    -/
    exact AEStronglyMeasurable.piecewise hs hf.1 hg.1
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    g : α → E
    hs : MeasurableSet s
    hf : MeasureTheory.Memℒp f p (μ.restrict s)
    hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
    hp_zero : Not (Eq p 0)
    ⊢ MeasureTheory.Memℒp (s.piecewise f g) p μ
  -/
  refine ⟨AEStronglyMeasurable.piecewise hs hf.1 hg.1, ?_⟩
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    g : α → E
    hs : MeasurableSet s
    hf : MeasureTheory.Memℒp f p (μ.restrict s)
    hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
    hp_zero : Not (Eq p 0)
    ⊢ LT.lt (MeasureTheory.eLpNorm (s.piecewise f g) p μ) Top.top
  -/
  rcases eq_or_ne p ∞ with rfl | hp_top
    /-
      case neg.inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f Top.top (μ.restrict s)
      hg : MeasureTheory.Memℒp g Top.top (μ.restrict (HasCompl.compl s))
      hp_zero : Not (Eq Top.top 0)
      ⊢ LT.lt (MeasureTheory.eLpNorm (s.piecewise f g) Top.top μ) Top.top
    -/
  · rw [eLpNorm_top_piecewise f g hs]
    /-
      case neg.inl
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f Top.top (μ.restrict s)
      hg : MeasureTheory.Memℒp g Top.top (μ.restrict (HasCompl.compl s))
      hp_zero : Not (Eq Top.top 0)
      ⊢ LT.lt (Max.max (MeasureTheory.eLpNorm f Top.top (μ.restrict s)) (MeasureTheo …
    -/
    exact max_lt hf.2 hg.2
    /-
      🎉 no goals
    -/
  rw [eLpNorm_lt_top_iff_lintegral_rpow_nnnorm_lt_top hp_zero hp_top, ← lintegral_add_compl _ hs,
    ENNReal.add_lt_top]
  /-
    case neg.inr
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    f : α → E
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    g : α → E
    hs : MeasurableSet s
    hf : MeasureTheory.Memℒp f p (μ.restrict s)
    hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
    hp_zero : Not (Eq p 0)
    hp_top : Ne p Top.top
    ⊢ And (LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => HPow.hPow (↑(NNN …
  -/
  constructor
  · have h : ∀ᵐ (x : α) ∂μ, x ∈ s →
        (‖Set.piecewise s f g x‖₊ : ℝ≥0∞) ^ p.toReal = (‖f x‖₊ : ℝ≥0∞) ^ p.toReal := by
      filter_upwards with a ha using by simp [ha]
    /-
      case neg.inr.left
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f p (μ.restrict s)
      hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
      hp_zero : Not (Eq p 0)
      hp_top : Ne p Top.top
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (HPow.hPow (↑(NNNorm.n …
      ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => HPow.hPow (↑(NNNorm.n …
    -/
    rw [setLIntegral_congr_fun hs h]
    /-
      case neg.inr.left
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f p (μ.restrict s)
      hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
      hp_zero : Not (Eq p 0)
      hp_top : Ne p Top.top
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (HPow.hPow (↑(NNNorm.n …
      ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => HPow.hPow (↑(NNNorm.n …
    -/
    exact lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top hp_zero hp_top hf.2
    /-
      🎉 no goals
    -/
  · have h : ∀ᵐ (x : α) ∂μ, x ∈ sᶜ →
        (‖Set.piecewise s f g x‖₊ : ℝ≥0∞) ^ p.toReal = (‖g x‖₊ : ℝ≥0∞) ^ p.toReal := by
      filter_upwards with a ha
      have ha' : a ∉ s := ha
      simp [ha']
    /-
      case neg.inr.right
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f p (μ.restrict s)
      hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
      hp_zero : Not (Eq p 0)
      hp_top : Ne p Top.top
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (HPow …
      ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun x => HPow …
    -/
    rw [setLIntegral_congr_fun hs.compl h]
    /-
      case neg.inr.right
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      f : α → E
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      g : α → E
      hs : MeasurableSet s
      hf : MeasureTheory.Memℒp f p (μ.restrict s)
      hg : MeasureTheory.Memℒp g p (μ.restrict (HasCompl.compl s))
      hp_zero : Not (Eq p 0)
      hp_top : Ne p Top.top
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (HPow …
      ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun x => HPow …
    -/
    exact lintegral_rpow_nnnorm_lt_top_of_eLpNorm_lt_top hp_zero hp_top hg.2
    /-
      🎉 no goals
    -/


/-- A bounded measurable function with compact support is in L^p. -/
theorem _root_.HasCompactSupport.memℒp_of_bound {f : X → E} (hf : HasCompactSupport f)
    (h2f : AEStronglyMeasurable f μ) (C : ℝ) (hfC : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) : Memℒp f p μ := by
  /-
    E : Type u_2
    p : ENNReal
    inst✝³ : NormedAddCommGroup E
    X : Type u_5
    inst✝² : TopologicalSpace X
    inst✝¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    f : X → E
    hf : HasCompactSupport f
    h2f : MeasureTheory.AEStronglyMeasurable f μ
    C : Real
    hfC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae …
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  have := memℒp_top_of_bound h2f C hfC
  exact this.memℒp_of_exponent_le_of_measure_support_ne_top
    (fun x ↦ image_eq_zero_of_nmem_tsupport) (hf.measure_lt_top.ne) le_top


/-- A continuous function with compact support is in L^p. -/
theorem _root_.Continuous.memℒp_of_hasCompactSupport [OpensMeasurableSpace X]
    {f : X → E} (hf : Continuous f) (h'f : HasCompactSupport f) : Memℒp f p μ := by
  /-
    E : Type u_2
    p : ENNReal
    inst✝⁴ : NormedAddCommGroup E
    X : Type u_5
    inst✝³ : TopologicalSpace X
    inst✝² : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : OpensMeasurableSpace X
    f : X → E
    hf : Continuous f
    h'f : HasCompactSupport f
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  have := hf.memℒp_top_of_hasCompactSupport h'f μ
  exact this.memℒp_of_exponent_le_of_measure_support_ne_top
    (fun x ↦ image_eq_zero_of_nmem_tsupport) (h'f.measure_lt_top.ne) le_top


/-- Indicator of a set as an element of `Lp`. -/
def indicatorConstLp (p : ℝ≥0∞) (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : E) : Lp E p μ :=
  Memℒp.toLp (s.indicator fun _ => c) (memℒp_indicator_const p hs c (Or.inr hμs))


/-- A version of `Set.indicator_add` for `MeasureTheory.indicatorConstLp`.-/
theorem indicatorConstLp_add {c' : E} :
    indicatorConstLp p hs hμs c + indicatorConstLp p hs hμs c' =
    indicatorConstLp p hs hμs (c + c') := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c c' : E
    ⊢ Eq (HAdd.hAdd (MeasureTheory.indicatorConstLp p hs hμs c) (MeasureTheory.ind …
  -/
  simp_rw [indicatorConstLp, ← Memℒp.toLp_add, indicator_add]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c c' : E
    ⊢ Eq (MeasureTheory.Memℒp.toLp (HAdd.hAdd (s.indicator fun x => c) (s.indicato …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A version of `Set.indicator_sub` for `MeasureTheory.indicatorConstLp`.-/
theorem indicatorConstLp_sub {c' : E} :
    indicatorConstLp p hs hμs c - indicatorConstLp p hs hμs c' =
    indicatorConstLp p hs hμs (c - c') := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c c' : E
    ⊢ Eq (HSub.hSub (MeasureTheory.indicatorConstLp p hs hμs c) (MeasureTheory.ind …
  -/
  simp_rw [indicatorConstLp, ← Memℒp.toLp_sub, indicator_sub]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c c' : E
    ⊢ Eq (MeasureTheory.Memℒp.toLp (HSub.hSub (s.indicator fun x => c) (s.indicato …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem indicatorConstLp_coeFn : ⇑(indicatorConstLp p hs hμs c) =ᵐ[μ] s.indicator fun _ => c :=
  Memℒp.coeFn_toLp (memℒp_indicator_const p hs c (Or.inr hμs))


theorem indicatorConstLp_coeFn_mem : ∀ᵐ x : α ∂μ, x ∈ s → indicatorConstLp p hs hμs c x = c :=
  indicatorConstLp_coeFn.mono fun _x hx hxs => hx.trans (Set.indicator_of_mem hxs _)


theorem indicatorConstLp_coeFn_nmem : ∀ᵐ x : α ∂μ, x ∉ s → indicatorConstLp p hs hμs c x = 0 :=
  indicatorConstLp_coeFn.mono fun _x hx hxs => hx.trans (Set.indicator_of_not_mem hxs _)


theorem norm_indicatorConstLp (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    ‖indicatorConstLp p hs hμs c‖ = ‖c‖ * (μ s).toReal ^ (1 / p.toReal) := by
  rw [Lp.norm_def, eLpNorm_congr_ae indicatorConstLp_coeFn,
    eLpNorm_indicator_const hs hp_ne_zero hp_ne_top, ENNReal.toReal_mul, ENNReal.toReal_rpow,
    ENNReal.coe_toReal, coe_nnnorm]


theorem norm_indicatorConstLp_top (hμs_ne_zero : μ s ≠ 0) :
    ‖indicatorConstLp ∞ hs hμs c‖ = ‖c‖ := by
  rw [Lp.norm_def, eLpNorm_congr_ae indicatorConstLp_coeFn,
    eLpNorm_indicator_const' hs hμs_ne_zero ENNReal.top_ne_zero, ENNReal.top_toReal,
    _root_.div_zero, ENNReal.rpow_zero, mul_one, ENNReal.coe_toReal, coe_nnnorm]


theorem norm_indicatorConstLp' (hp_pos : p ≠ 0) (hμs_pos : μ s ≠ 0) :
    ‖indicatorConstLp p hs hμs c‖ = ‖c‖ * (μ s).toReal ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    hp_pos : Ne p 0
    hμs_pos : Ne (μ s) 0
    ⊢ Eq (Norm.norm (MeasureTheory.indicatorConstLp p hs hμs c)) (HMul.hMul (Norm. …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      c : E
      hp_pos : Ne p 0
      hμs_pos : Ne (μ s) 0
      hp_top : Eq p Top.top
      ⊢ Eq (Norm.norm (MeasureTheory.indicatorConstLp p hs hμs c)) (HMul.hMul (Norm. …
    -/
  · rw [hp_top, ENNReal.top_toReal, _root_.div_zero, Real.rpow_zero, mul_one]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      c : E
      hp_pos : Ne p 0
      hμs_pos : Ne (μ s) 0
      hp_top : Eq p Top.top
      ⊢ Eq (Norm.norm (MeasureTheory.indicatorConstLp Top.top hs hμs c)) (Norm.norm c)
    -/
    exact norm_indicatorConstLp_top hμs_pos
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      c : E
      hp_pos : Ne p 0
      hμs_pos : Ne (μ s) 0
      hp_top : Not (Eq p Top.top)
      ⊢ Eq (Norm.norm (MeasureTheory.indicatorConstLp p hs hμs c)) (HMul.hMul (Norm. …
    -/
  · exact norm_indicatorConstLp hp_pos hp_top
    /-
      🎉 no goals
    -/


theorem norm_indicatorConstLp_le :
    ‖indicatorConstLp p hs hμs c‖ ≤ ‖c‖ * (μ s).toReal ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ LE.le (Norm.norm (MeasureTheory.indicatorConstLp p hs hμs c)) (HMul.hMul (No …
  -/
  rw [indicatorConstLp, Lp.norm_toLp]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ).toReal (HMul.hMul …
  -/
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ LE.le (MeasureTheory.eLpNorm (s.indicator fun x => c) p μ) (ENNReal.ofReal ( …
  -/
  refine (eLpNorm_indicator_const_le _ _).trans_eq ?_
  rw [← coe_nnnorm, ENNReal.ofReal_mul (NNReal.coe_nonneg _), ENNReal.ofReal_coe_nnreal,
    ENNReal.toReal_rpow, ENNReal.ofReal_toReal]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ Ne (HPow.hPow (μ s) (HDiv.hDiv 1 p.toReal)) Top.top
  -/
  exact ENNReal.rpow_ne_top_of_nonneg (by positivity) hμs
  /-
    🎉 no goals
  -/


theorem nnnorm_indicatorConstLp_le :
    ‖indicatorConstLp p hs hμs c‖₊ ≤ ‖c‖₊ * (μ s).toNNReal ^ (1 / p.toReal) :=
  norm_indicatorConstLp_le


theorem ennnorm_indicatorConstLp_le :
    (‖indicatorConstLp p hs hμs c‖₊ : ℝ≥0∞) ≤ ‖c‖₊ * (μ s) ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ LE.le (↑(NNNorm.nnnorm (MeasureTheory.indicatorConstLp p hs hμs c))) (HMul.h …
  -/
  refine (ENNReal.coe_le_coe.mpr nnnorm_indicatorConstLp_le).trans_eq ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ Eq (↑(HMul.hMul (NNNorm.nnnorm c) (HPow.hPow (μ s).toNNReal (HDiv.hDiv 1 p.t …
  -/
  simp [ENNReal.coe_rpow_of_nonneg, ENNReal.coe_toNNReal hμs]
  /-
    🎉 no goals
  -/


theorem edist_indicatorConstLp_eq_nnnorm {t : Set α} {ht : MeasurableSet t} {hμt : μ t ≠ ∞} :
    edist (indicatorConstLp p hs hμs c) (indicatorConstLp p ht hμt c) =
      ‖indicatorConstLp p (hs.symmDiff ht) (measure_symmDiff_ne_top hμs hμt) c‖₊ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ Eq (EDist.edist (MeasureTheory.indicatorConstLp p hs hμs c) (MeasureTheory.i …
  -/
  unfold indicatorConstLp
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ Eq (EDist.edist (MeasureTheory.Memℒp.toLp (s.indicator fun x => c) ⋯) (Measu …
  -/
  rw [Lp.edist_toLp_toLp, eLpNorm_indicator_sub_indicator, Lp.coe_nnnorm_toLp]
  /-
    🎉 no goals
  -/


theorem dist_indicatorConstLp_eq_norm {t : Set α} {ht : MeasurableSet t} {hμt : μ t ≠ ∞} :
    dist (indicatorConstLp p hs hμs c) (indicatorConstLp p ht hμt c) =
      ‖indicatorConstLp p (hs.symmDiff ht) (measure_symmDiff_ne_top hμs hμt) c‖ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ Eq (Dist.dist (MeasureTheory.indicatorConstLp p hs hμs c) (MeasureTheory.ind …
  -/
  rw [Lp.dist_edist, edist_indicatorConstLp_eq_nnnorm, ENNReal.coe_toReal, Lp.coe_nnnorm]
  /-
    🎉 no goals
  -/


/-- A family of `indicatorConstLp` functions tends to an `indicatorConstLp`,
if the underlying sets tend to the set in the sense of the measure of the symmetric difference. -/
theorem tendsto_indicatorConstLp_set [hp₁ : Fact (1 ≤ p)] {β : Type*} {l : Filter β} {t : β → Set α}
    {ht : ∀ b, MeasurableSet (t b)} {hμt : ∀ b, μ (t b) ≠ ∞} (hp : p ≠ ∞)
    (h : Tendsto (fun b ↦ μ (t b ∆ s)) l (𝓝 0)) :
    Tendsto (fun b ↦ indicatorConstLp p (ht b) (hμt b) c) l (𝓝 (indicatorConstLp p hs hμs c)) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    hp₁ : Fact (LE.le 1 p)
    β : Type u_5
    l : Filter β
    t : β → Set α
    ht : ∀ (b : β), MeasurableSet (t b)
    hμt : ∀ (b : β), Ne (μ (t b)) Top.top
    hp : Ne p Top.top
    h : Filter.Tendsto (fun b => μ (symmDiff (t b) s)) l (nhds 0)
    ⊢ Filter.Tendsto (fun b => MeasureTheory.indicatorConstLp p ⋯ ⋯ c) l (nhds (Me …
  -/
  rw [tendsto_iff_dist_tendsto_zero]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    hp₁ : Fact (LE.le 1 p)
    β : Type u_5
    l : Filter β
    t : β → Set α
    ht : ∀ (b : β), MeasurableSet (t b)
    hμt : ∀ (b : β), Ne (μ (t b)) Top.top
    hp : Ne p Top.top
    h : Filter.Tendsto (fun b => μ (symmDiff (t b) s)) l (nhds 0)
    ⊢ Filter.Tendsto (fun b => Dist.dist (MeasureTheory.indicatorConstLp p ⋯ ⋯ c)  …
  -/
  have hp₀ : p ≠ 0 := (one_pos.trans_le hp₁.out).ne'
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    hp₁ : Fact (LE.le 1 p)
    β : Type u_5
    l : Filter β
    t : β → Set α
    ht : ∀ (b : β), MeasurableSet (t b)
    hμt : ∀ (b : β), Ne (μ (t b)) Top.top
    hp : Ne p Top.top
    h : Filter.Tendsto (fun b => μ (symmDiff (t b) s)) l (nhds 0)
    hp₀ : Ne p 0
    ⊢ Filter.Tendsto (fun b => Dist.dist (MeasureTheory.indicatorConstLp p ⋯ ⋯ c)  …
  -/
  simp only [dist_indicatorConstLp_eq_norm, norm_indicatorConstLp hp₀ hp]
  convert tendsto_const_nhds.mul
    (((ENNReal.tendsto_toReal ENNReal.zero_ne_top).comp h).rpow_const _)
    /-
      case h.e'_5.h.e'_3
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      c : E
      hp₁ : Fact (LE.le 1 p)
      β : Type u_5
      l : Filter β
      t : β → Set α
      ht : ∀ (b : β), MeasurableSet (t b)
      hμt : ∀ (b : β), Ne (μ (t b)) Top.top
      hp : Ne p Top.top
      h : Filter.Tendsto (fun b => μ (symmDiff (t b) s)) l (nhds 0)
      hp₀ : Ne p 0
      ⊢ Eq 0 (HMul.hMul (Norm.norm c) (HPow.hPow (ENNReal.toReal 0) (HDiv.hDiv 1 p.t …
    -/
  · simp [Real.rpow_eq_zero_iff_of_nonneg, ENNReal.toReal_eq_zero_iff, hp, hp₀]
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      s : Set α
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      c : E
      hp₁ : Fact (LE.le 1 p)
      β : Type u_5
      l : Filter β
      t : β → Set α
      ht : ∀ (b : β), MeasurableSet (t b)
      hμt : ∀ (b : β), Ne (μ (t b)) Top.top
      hp : Ne p Top.top
      h : Filter.Tendsto (fun b => μ (symmDiff (t b) s)) l (nhds 0)
      hp₀ : Ne p 0
      ⊢ Or (Ne (ENNReal.toReal 0) 0) (LE.le 0 (HDiv.hDiv 1 p.toReal))
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- A family of `indicatorConstLp` functions is continuous in the parameter,
if `μ (s y ∆ s x)` tends to zero as `y` tends to `x` for all `x`. -/
theorem continuous_indicatorConstLp_set [Fact (1 ≤ p)] {X : Type*} [TopologicalSpace X]
    {s : X → Set α} {hs : ∀ x, MeasurableSet (s x)} {hμs : ∀ x, μ (s x) ≠ ∞} (hp : p ≠ ∞)
    (h : ∀ x, Tendsto (fun y ↦ μ (s y ∆ s x)) (𝓝 x) (𝓝 0)) :
    Continuous fun x ↦ indicatorConstLp p (hs x) (hμs x) c :=
  continuous_iff_continuousAt.2 fun x ↦ tendsto_indicatorConstLp_set hp (h x)


@[simp]
theorem indicatorConstLp_empty :
                                               /-
                                                 α : Type u_1
                                                 E : Type u_2
                                                 F : Type u_3
                                                 G : Type u_4
                                                 m m0 : MeasurableSpace α
                                                 p : ENNReal
                                                 q : Real
                                                 μ ν : MeasureTheory.Measure α
                                                 inst✝² : NormedAddCommGroup E
                                                 inst✝¹ : NormedAddCommGroup F
                                                 inst✝ : NormedAddCommGroup G
                                                 s : Set α
                                                 hs : MeasurableSet s
                                                 hμs : Ne (μ s) Top.top
                                                 c : E
                                                 ⊢ Ne (μ EmptyCollection.emptyCollection) Top.top
                                               -/
    indicatorConstLp p MeasurableSet.empty (by simp : μ ∅ ≠ ∞) c = 0 := by
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    c : E
    ⊢ Eq (MeasureTheory.indicatorConstLp p ⋯ ⋯ c) 0
  -/
  simp only [indicatorConstLp, Set.indicator_empty', Memℒp.toLp_zero]
  /-
    🎉 no goals
  -/


theorem indicatorConstLp_inj {s t : Set α} (hs : MeasurableSet s) (hsμ : μ s ≠ ∞)
    (ht : MeasurableSet t) (htμ : μ t ≠ ∞) {c : E} (hc : c ≠ 0) :
    indicatorConstLp p hs hsμ c = indicatorConstLp p ht htμ c ↔ s =ᵐ[μ] t := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s t : Set α
    hs : MeasurableSet s
    hsμ : Ne (μ s) Top.top
    ht : MeasurableSet t
    htμ : Ne (μ t) Top.top
    c : E
    hc : Ne c 0
    ⊢ Iff (Eq (MeasureTheory.indicatorConstLp p hs hsμ c) (MeasureTheory.indicator …
  -/
  simp_rw [← indicator_const_eventuallyEq hc, indicatorConstLp, Memℒp.toLp_eq_toLp_iff]
  /-
    🎉 no goals
  -/


theorem memℒp_add_of_disjoint {f g : α → E} (h : Disjoint (support f) (support g))
    (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) :
    Memℒp (f + g) p μ ↔ Memℒp f p μ ∧ Memℒp g p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : α → E
    h : Disjoint (Function.support f) (Function.support g)
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ Iff (MeasureTheory.Memℒp (HAdd.hAdd f g) p μ) (And (MeasureTheory.Memℒp f p  …
  -/
  borelize E
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f g : α → E
    h : Disjoint (Function.support f) (Function.support g)
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ Iff (MeasureTheory.Memℒp (HAdd.hAdd f g) p μ) (And (MeasureTheory.Memℒp f p  …
  -/
  refine ⟨fun hfg => ⟨?_, ?_⟩, fun h => h.1.add h.2⟩
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f g : α → E
      h : Disjoint (Function.support f) (Function.support g)
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.StronglyMeasurable g
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      hfg : MeasureTheory.Memℒp (HAdd.hAdd f g) p μ
      ⊢ MeasureTheory.Memℒp f p μ
    -/
  · rw [← Set.indicator_add_eq_left h]; exact hfg.indicator (measurableSet_support hf.measurable)
                                        /-
                                          🎉 no goals
                                        -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f g : α → E
      h : Disjoint (Function.support f) (Function.support g)
      hf : MeasureTheory.StronglyMeasurable f
      hg : MeasureTheory.StronglyMeasurable g
      this✝¹ : MeasurableSpace E := borel E
      this✝ : BorelSpace E
      hfg : MeasureTheory.Memℒp (HAdd.hAdd f g) p μ
      ⊢ MeasureTheory.Memℒp g p μ
    -/
  · rw [← Set.indicator_add_eq_right h]; exact hfg.indicator (measurableSet_support hg.measurable)
                                         /-
                                           🎉 no goals
                                         -/


/-- The indicator of a disjoint union of two sets is the sum of the indicators of the sets. -/
theorem indicatorConstLp_disjoint_union {s t : Set α} (hs : MeasurableSet s) (ht : MeasurableSet t)
    (hμs : μ s ≠ ∞) (hμt : μ t ≠ ∞) (hst : Disjoint s t) (c : E) :
    indicatorConstLp p (hs.union ht) (measure_union_ne_top hμs hμt) c =
      indicatorConstLp p hs hμs c + indicatorConstLp p ht hμt c := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    c : E
    ⊢ Eq (MeasureTheory.indicatorConstLp p ⋯ ⋯ c) (HAdd.hAdd (MeasureTheory.indica …
  -/
  ext1
  /-
    case h
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    c : E
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.indicatorConstLp p ⋯ ⋯ c) …
  -/
  refine indicatorConstLp_coeFn.trans (EventuallyEq.trans ?_ (Lp.coeFn_add _ _).symm)
  refine
    EventuallyEq.trans ?_
      (EventuallyEq.add indicatorConstLp_coeFn.symm indicatorConstLp_coeFn.symm)
  /-
    case h
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    c : E
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((Union.union s t).indicator fun x => c) f …
  -/
  rw [Set.indicator_union_of_disjoint hst]
  /-
    🎉 no goals
  -/


/-- Constant function as an element of `MeasureTheory.Lp` for a finite measure. -/
protected def Lp.const : E →+ Lp E p μ where
  toFun c := ⟨AEEqFun.const α c, const_mem_Lp α μ c⟩
  map_zero' := rfl
  map_add' _ _ := rfl


lemma Lp.coeFn_const : Lp.const p μ c =ᵐ[μ] Function.const α c :=
  AEEqFun.coeFn_const α c


@[simp] lemma Lp.const_val : (Lp.const p μ c).1 = AEEqFun.const α c := rfl


@[simp]
lemma Memℒp.toLp_const : Memℒp.toLp _ (memℒp_const c) = Lp.const p μ c := rfl


@[simp]
lemma indicatorConstLp_univ :
    indicatorConstLp p .univ (measure_ne_top μ _) c = Lp.const p μ c := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    ⊢ Eq (MeasureTheory.indicatorConstLp p ⋯ ⋯ c) ((MeasureTheory.Lp.const p μ) c)
  -/
  rw [← Memℒp.toLp_const, indicatorConstLp]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    ⊢ Eq (MeasureTheory.Memℒp.toLp (Set.univ.indicator fun x => c) ⋯) (MeasureTheo …
  -/
  simp only [Set.indicator_univ, Function.const]
  /-
    🎉 no goals
  -/


theorem Lp.norm_const [NeZero μ] (hp_zero : p ≠ 0) :
    ‖Lp.const p μ c‖ = ‖c‖ * (μ Set.univ).toReal ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    c : E
    inst✝ : NeZero μ
    hp_zero : Ne p 0
    ⊢ Eq (Norm.norm ((MeasureTheory.Lp.const p μ) c)) (HMul.hMul (Norm.norm c) (HP …
  -/
  have := NeZero.ne μ
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    c : E
    inst✝ : NeZero μ
    hp_zero : Ne p 0
    this : Ne μ 0
    ⊢ Eq (Norm.norm ((MeasureTheory.Lp.const p μ) c)) (HMul.hMul (Norm.norm c) (HP …
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  rw [← Memℒp.toLp_const, Lp.norm_toLp, eLpNorm_const] <;> try assumption
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    c : E
    inst✝ : NeZero μ
    hp_zero : Ne p 0
    this : Ne μ 0
    ⊢ Eq (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDiv 1 p.to …
  -/
  rw [ENNReal.toReal_mul, ENNReal.coe_toReal, ← ENNReal.toReal_rpow, coe_nnnorm]
  /-
    🎉 no goals
  -/


theorem Lp.norm_const' (hp_zero : p ≠ 0) (hp_top : p ≠ ∞) :
    ‖Lp.const p μ c‖ = ‖c‖ * (μ Set.univ).toReal ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    hp_zero : Ne p 0
    hp_top : Ne p Top.top
    ⊢ Eq (Norm.norm ((MeasureTheory.Lp.const p μ) c)) (HMul.hMul (Norm.norm c) (HP …
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  rw [← Memℒp.toLp_const, Lp.norm_toLp, eLpNorm_const'] <;> try assumption
                                                            /-
                                                              🎉 no goals
                                                            -/
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    hp_zero : Ne p 0
    hp_top : Ne p Top.top
    ⊢ Eq (HMul.hMul (↑(NNNorm.nnnorm c)) (HPow.hPow (μ Set.univ) (HDiv.hDiv 1 p.to …
  -/
  rw [ENNReal.toReal_mul, ENNReal.coe_toReal, ← ENNReal.toReal_rpow, coe_nnnorm]
  /-
    🎉 no goals
  -/


theorem Lp.norm_const_le : ‖Lp.const p μ c‖ ≤ ‖c‖ * (μ Set.univ).toReal ^ (1 / p.toReal) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    ⊢ LE.le (Norm.norm ((MeasureTheory.Lp.const p μ) c)) (HMul.hMul (Norm.norm c)  …
  -/
  rw [← indicatorConstLp_univ]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    ⊢ LE.le (Norm.norm (MeasureTheory.indicatorConstLp p ⋯ ⋯ c)) (HMul.hMul (Norm. …
  -/
  exact norm_indicatorConstLp_le
  /-
    🎉 no goals
  -/


/-- `MeasureTheory.Lp.const` as a `LinearMap`. -/
@[simps] protected def Lp.constₗ (𝕜 : Type*) [NormedRing 𝕜] [Module 𝕜 E] [BoundedSMul 𝕜 E] :
    E →ₗ[𝕜] Lp E p μ where
  toFun := Lp.const p μ
  map_add' := map_add _
  map_smul' _ _ := rfl


@[simps! apply]
protected def Lp.constL (𝕜 : Type*) [NormedField 𝕜] [NormedSpace 𝕜 E] [Fact (1 ≤ p)] :
    E →L[𝕜] Lp E p μ :=
  (Lp.constₗ p μ 𝕜).mkContinuous ((μ Set.univ).toReal ^ (1 / p.toReal)) fun _ ↦
    (Lp.norm_const_le _ _ _).trans_eq (mul_comm _ _)


theorem Lp.norm_constL_le (𝕜 : Type*) [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E]
    [Fact (1 ≤ p)] :
    ‖(Lp.constL p μ 𝕜 : E →L[𝕜] Lp E p μ)‖ ≤ (μ Set.univ).toReal ^ (1 / p.toReal) :=
                                       /-
                                         α : Type u_1
                                         E : Type u_2
                                         m0 : MeasurableSpace α
                                         p : ENNReal
                                         μ : MeasureTheory.Measure α
                                         inst✝⁴ : NormedAddCommGroup E
                                         inst✝³ : MeasureTheory.IsFiniteMeasure μ
                                         𝕜 : Type u_5
                                         inst✝² : NontriviallyNormedField 𝕜
                                         inst✝¹ : NormedSpace 𝕜 E
                                         inst✝ : Fact (LE.le 1 p)
                                         ⊢ LE.le 0 (HPow.hPow (μ Set.univ).toReal (HDiv.hDiv 1 p.toReal))
                                       -/
  LinearMap.mkContinuous_norm_le _ (by positivity) _
                                       /-
                                         🎉 no goals
                                       -/


theorem Memℒp.norm_rpow_div {f : α → E} (hf : Memℒp f p μ) (q : ℝ≥0∞) :
    Memℒp (fun x : α => ‖f x‖ ^ q.toReal) (p / q) μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    q : ENNReal
    ⊢ MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) (HDiv.hD …
  -/
  refine ⟨(hf.1.norm.aemeasurable.pow_const q.toReal).aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    q : ENNReal
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) …
  -/
  by_cases q_top : q = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      q : ENNReal
      q_top : Eq q Top.top
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) …
    -/
  · simp [q_top]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    q : ENNReal
    q_top : Not (Eq q Top.top)
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) …
  -/
  by_cases q_zero : q = 0
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      q : ENNReal
      q_top : Not (Eq q Top.top)
      q_zero : Eq q 0
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) …
    -/
  · simp only [q_zero, ENNReal.zero_toReal, Real.rpow_zero]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      q : ENNReal
      q_top : Not (Eq q Top.top)
      q_zero : Eq q 0
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => 1) (HDiv.hDiv p 0) μ) Top.top
    -/
    by_cases p_zero : p = 0
      /-
        case pos
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup E
        f : α → E
        hf : MeasureTheory.Memℒp f p μ
        q : ENNReal
        q_top : Not (Eq q Top.top)
        q_zero : Eq q 0
        p_zero : Eq p 0
        ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => 1) (HDiv.hDiv p 0) μ) Top.top
      -/
    · simp [p_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      q : ENNReal
      q_top : Not (Eq q Top.top)
      q_zero : Eq q 0
      p_zero : Not (Eq p 0)
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => 1) (HDiv.hDiv p 0) μ) Top.top
    -/
    rw [ENNReal.div_zero p_zero]
    /-
      case neg
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : α → E
      hf : MeasureTheory.Memℒp f p μ
      q : ENNReal
      q_top : Not (Eq q Top.top)
      q_zero : Eq q 0
      p_zero : Not (Eq p 0)
      ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => 1) Top.top μ) Top.top
    -/
    exact (memℒp_top_const (1 : ℝ)).2
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    q : ENNReal
    q_top : Not (Eq q Top.top)
    q_zero : Not (Eq q 0)
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) …
  -/
  rw [eLpNorm_norm_rpow _ (ENNReal.toReal_pos q_zero q_top)]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    q : ENNReal
    q_top : Not (Eq q Top.top)
    q_zero : Not (Eq q 0)
    ⊢ LT.lt (HPow.hPow (MeasureTheory.eLpNorm f (HMul.hMul (HDiv.hDiv p q) (ENNRea …
  -/
  apply ENNReal.rpow_lt_top_of_nonneg ENNReal.toReal_nonneg
  rw [ENNReal.ofReal_toReal q_top, div_eq_mul_inv, mul_assoc, ENNReal.inv_mul_cancel q_zero q_top,
    mul_one]
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    q : ENNReal
    q_top : Not (Eq q Top.top)
    q_zero : Not (Eq q 0)
    ⊢ Ne (MeasureTheory.eLpNorm f p μ) Top.top
  -/
  exact hf.2.ne
  /-
    🎉 no goals
  -/


theorem memℒp_norm_rpow_iff {q : ℝ≥0∞} {f : α → E} (hf : AEStronglyMeasurable f μ) (q_zero : q ≠ 0)
    (q_top : q ≠ ∞) : Memℒp (fun x : α => ‖f x‖ ^ q.toReal) (p / q) μ ↔ Memℒp f p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    q : ENNReal
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    q_zero : Ne q 0
    q_top : Ne q Top.top
    ⊢ Iff (MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) (HD …
  -/
  refine ⟨fun h => ?_, fun h => h.norm_rpow_div q⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    q : ENNReal
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    q_zero : Ne q 0
    q_top : Ne q Top.top
    h : MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) (HDiv. …
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  apply (memℒp_norm_iff hf).1
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    q : ENNReal
    f : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    q_zero : Ne q 0
    q_top : Ne q Top.top
    h : MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) (HDiv. …
    ⊢ MeasureTheory.Memℒp (fun x => Norm.norm (f x)) p μ
  -/
  convert h.norm_rpow_div q⁻¹ using 1
    /-
      case h.e'_6
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      q : ENNReal
      f : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      q_zero : Ne q 0
      q_top : Ne q Top.top
      h : MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) (HDiv. …
      ⊢ Eq (fun x => Norm.norm (f x)) fun x => HPow.hPow (Norm.norm (HPow.hPow (Norm …
    -/
  · ext x
    rw [Real.norm_eq_abs, Real.abs_rpow_of_nonneg (norm_nonneg _), ← Real.rpow_mul (abs_nonneg _),
      ENNReal.toReal_inv, mul_inv_cancel₀, abs_of_nonneg (norm_nonneg _), Real.rpow_one]
    /-
      case h.e'_6.h
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      q : ENNReal
      f : α → E
      hf : MeasureTheory.AEStronglyMeasurable f μ
      q_zero : Ne q 0
      q_top : Ne q Top.top
      h : MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) q.toReal) (HDiv. …
      x : α
      ⊢ Ne q.toReal 0
    -/
    simp [ENNReal.toReal_eq_zero_iff, not_or, q_zero, q_top]
    /-
      🎉 no goals
    -/
  · rw [div_eq_mul_inv, inv_inv, div_eq_mul_inv, mul_assoc, ENNReal.inv_mul_cancel q_zero q_top,
      mul_one]


theorem Memℒp.norm_rpow {f : α → E} (hf : Memℒp f p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) :
    Memℒp (fun x : α => ‖f x‖ ^ p.toReal) 1 μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) 1 μ
  -/
  convert hf.norm_rpow_div p
  /-
    case h.e'_7
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : α → E
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ Eq 1 (HDiv.hDiv p p)
  -/
  rw [div_eq_mul_inv, ENNReal.mul_inv_cancel hp_ne_zero hp_ne_top]
  /-
    🎉 no goals
  -/


theorem AEEqFun.compMeasurePreserving_mem_Lp {β : Type*} [MeasurableSpace β]
    {μb : MeasureTheory.Measure β} {g : β →ₘ[μb] E} (hg : g ∈ Lp E p μb) {f : α → β}
    (hf : MeasurePreserving f μ μb) :
    g.compMeasurePreserving f hf ∈ Lp E p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    β : Type u_5
    inst✝ : MeasurableSpace β
    μb : MeasureTheory.Measure β
    g : MeasureTheory.AEEqFun β E μb
    hg : Membership.mem (MeasureTheory.Lp E p μb) g
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μ μb
    ⊢ Membership.mem (MeasureTheory.Lp E p μ) (g.compMeasurePreserving f hf)
  -/
  rw [Lp.mem_Lp_iff_eLpNorm_lt_top] at hg ⊢
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    β : Type u_5
    inst✝ : MeasurableSpace β
    μb : MeasureTheory.Measure β
    g : MeasureTheory.AEEqFun β E μb
    hg : LT.lt (MeasureTheory.eLpNorm (↑g) p μb) Top.top
    f : α → β
    hf : MeasureTheory.MeasurePreserving f μ μb
    ⊢ LT.lt (MeasureTheory.eLpNorm (↑(g.compMeasurePreserving f hf)) p μ) Top.top
  -/
  rwa [eLpNorm_compMeasurePreserving]
  /-
    🎉 no goals
  -/


/-- Composition of an `L^p` function with a measure preserving function is an `L^p` function. -/
def compMeasurePreserving (f : α → β) (hf : MeasurePreserving f μ μb) :
    Lp E p μb →+ Lp E p μ where
  toFun g := ⟨g.1.compMeasurePreserving f hf, g.1.compMeasurePreserving_mem_Lp g.2 hf⟩
  map_zero' := rfl
                 /-
                   α : Type u_1
                   E : Type u_2
                   F : Type u_3
                   G : Type u_4
                   m m0 : MeasurableSpace α
                   p : ENNReal
                   q : Real
                   μ ν : MeasureTheory.Measure α
                   inst✝³ : NormedAddCommGroup E
                   inst✝² : NormedAddCommGroup F
                   inst✝¹ : NormedAddCommGroup G
                   β : Type u_5
                   inst✝ : MeasurableSpace β
                   μb : MeasureTheory.Measure β
                   f✝ f : α → β
                   hf : MeasureTheory.MeasurePreserving f μ μb
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μb) x), Eq ({ …
                 -/
  map_add' := by rintro ⟨⟨_⟩, _⟩ ⟨⟨_⟩, _⟩; rfl
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem compMeasurePreserving_val (g : Lp E p μb) (hf : MeasurePreserving f μ μb) :
    (compMeasurePreserving f hf g).1 = g.1.compMeasurePreserving f hf :=
  rfl


theorem coeFn_compMeasurePreserving (g : Lp E p μb) (hf : MeasurePreserving f μ μb) :
    compMeasurePreserving f hf g =ᵐ[μ] g ∘ f :=
  g.1.coeFn_compMeasurePreserving hf


@[simp]
theorem norm_compMeasurePreserving (g : Lp E p μb) (hf : MeasurePreserving f μ μb) :
    ‖compMeasurePreserving f hf g‖ = ‖g‖ :=
  congr_arg ENNReal.toReal <| g.1.eLpNorm_compMeasurePreserving hf


theorem isometry_compMeasurePreserving [Fact (1 ≤ p)] (hf : MeasurePreserving f μ μb) :
    Isometry (compMeasurePreserving f hf : Lp E p μb → Lp E p μ) :=
  AddMonoidHomClass.isometry_of_norm _ (norm_compMeasurePreserving · hf)


theorem toLp_compMeasurePreserving {g : β → E} (hg : Memℒp g p μb) (hf : MeasurePreserving f μ μb) :
    compMeasurePreserving f hf (hg.toLp g) = (hg.comp_measurePreserving hf).toLp _ := rfl


theorem indicatorConstLp_compMeasurePreserving {s : Set β} (hs : MeasurableSet s)
    (hμs : μb s ≠ ∞) (c : E) (hf : MeasurePreserving f μ μb) :
    Lp.compMeasurePreserving f hf (indicatorConstLp p hs hμs c) =
      indicatorConstLp p (hs.preimage hf.measurable)
            /-
              α : Type u_1
              E : Type u_2
              F : Type u_3
              G : Type u_4
              m m0 : MeasurableSpace α
              p : ENNReal
              q : Real
              μ ν : MeasureTheory.Measure α
              inst✝³ : NormedAddCommGroup E
              inst✝² : NormedAddCommGroup F
              inst✝¹ : NormedAddCommGroup G
              β : Type u_5
              inst✝ : MeasurableSpace β
              μb : MeasureTheory.Measure β
              f : α → β
              s : Set β
              hs : MeasurableSet s
              hμs : Ne (μb s) Top.top
              c : E
              hf : MeasureTheory.MeasurePreserving f μ μb
              ⊢ Ne (μ (Set.preimage f s)) Top.top
            -/
        (by rwa [hf.measure_preimage hs.nullMeasurableSet]) c :=
            /-
              🎉 no goals
            -/
  rfl


/-- `MeasureTheory.Lp.compMeasurePreserving` as a linear map. -/
@[simps]
def compMeasurePreservingₗ (f : α → β) (hf : MeasurePreserving f μ μb) :
    Lp E p μb →ₗ[𝕜] Lp E p μ where
  __ := compMeasurePreserving f hf
                      /-
                        α : Type u_1
                        E : Type u_2
                        F : Type u_3
                        G : Type u_4
                        m m0 : MeasurableSpace α
                        p : ENNReal
                        q : Real
                        μ ν : MeasureTheory.Measure α
                        inst✝⁶ : NormedAddCommGroup E
                        inst✝⁵ : NormedAddCommGroup F
                        inst✝⁴ : NormedAddCommGroup G
                        β : Type u_5
                        inst✝³ : MeasurableSpace β
                        μb : MeasureTheory.Measure β
                        f✝ : α → β
                        𝕜 : Type u_6
                        inst✝² : NormedRing 𝕜
                        inst✝¹ : Module 𝕜 E
                        inst✝ : BoundedSMul 𝕜 E
                        f : α → β
                        hf : MeasureTheory.MeasurePreserving f μ μb
                        c : 𝕜
                        g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μb) x
                        ⊢ Eq ({ toFun := (↑__spread✝⁻⁰).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c g) …
                      -/
  map_smul' c g := by rcases g with ⟨⟨_⟩, _⟩; rfl
                                              /-
                                                🎉 no goals
                                              -/


/-- `MeasureTheory.Lp.compMeasurePreserving` as a linear isometry. -/
@[simps!]
def compMeasurePreservingₗᵢ [Fact (1 ≤ p)] (f : α → β) (hf : MeasurePreserving f μ μb) :
    Lp E p μb →ₗᵢ[𝕜] Lp E p μ where
  toLinearMap := compMeasurePreservingₗ 𝕜 f hf
  norm_map' := (norm_compMeasurePreserving · hf)


theorem LipschitzWith.comp_memℒp {α E F} {K} [MeasurableSpace α] {μ : Measure α}
    [NormedAddCommGroup E] [NormedAddCommGroup F] {f : α → E} {g : E → F} (hg : LipschitzWith K g)
    (g0 : g 0 = 0) (hL : Memℒp f p μ) : Memℒp (g ∘ f) p μ :=
  have : ∀ x, ‖g (f x)‖ ≤ K * ‖f x‖ := fun x ↦ by
    -- TODO: add `LipschitzWith.nnnorm_sub_le` and `LipschitzWith.nnnorm_le`
    /-
      p : ENNReal
      α : Type u_5
      E : Type u_6
      F : Type u_7
      K : NNReal
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedAddCommGroup F
      f : α → E
      g : E → F
      hg : LipschitzWith K g
      g0 : Eq (g 0) 0
      hL : MeasureTheory.Memℒp f p μ
      x : α
      ⊢ LE.le (Norm.norm (g (f x))) (HMul.hMul (↑K) (Norm.norm (f x)))
    -/
    simpa [g0] using hg.norm_sub_le (f x) 0
    /-
      🎉 no goals
    -/
  hL.of_le_mul (hg.continuous.comp_aestronglyMeasurable hL.1) (Eventually.of_forall this)


theorem MeasureTheory.Memℒp.of_comp_antilipschitzWith {α E F} {K'} [MeasurableSpace α]
    {μ : Measure α} [NormedAddCommGroup E] [NormedAddCommGroup F] {f : α → E} {g : E → F}
    (hL : Memℒp (g ∘ f) p μ) (hg : UniformContinuous g) (hg' : AntilipschitzWith K' g)
    (g0 : g 0 = 0) : Memℒp f p μ := by
  have A : ∀ x, ‖f x‖ ≤ K' * ‖g (f x)‖ := by
    intro x
    -- TODO: add `AntilipschitzWith.le_mul_nnnorm_sub` and `AntilipschitzWith.le_mul_norm`
    rw [← dist_zero_right, ← dist_zero_right, ← g0]
    apply hg'.le_mul_dist
  have B : AEStronglyMeasurable f μ :=
    (hg'.isUniformEmbedding hg).isEmbedding.aestronglyMeasurable_comp_iff.1 hL.1
  /-
    p : ENNReal
    α : Type u_5
    E : Type u_6
    F : Type u_7
    K' : NNReal
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : α → E
    g : E → F
    hL : MeasureTheory.Memℒp (Function.comp g f) p μ
    hg : UniformContinuous g
    hg' : AntilipschitzWith K' g
    g0 : Eq (g 0) 0
    A : ∀ (x : α), LE.le (Norm.norm (f x)) (HMul.hMul (↑K') (Norm.norm (g (f x))))
    B : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  exact hL.of_le_mul B (Filter.Eventually.of_forall A)
  /-
    🎉 no goals
  -/


theorem memℒp_comp_iff_of_antilipschitz {α E F} {K K'} [MeasurableSpace α] {μ : Measure α}
    [NormedAddCommGroup E] [NormedAddCommGroup F] {f : α → E} {g : E → F} (hg : LipschitzWith K g)
    (hg' : AntilipschitzWith K' g) (g0 : g 0 = 0) : Memℒp (g ∘ f) p μ ↔ Memℒp f p μ :=
  ⟨fun h => h.of_comp_antilipschitzWith hg.uniformContinuous hg' g0, fun h => hg.comp_memℒp g0 h⟩


/-- When `g` is a Lipschitz function sending `0` to `0` and `f` is in `Lp`, then `g ∘ f` is well
defined as an element of `Lp`. -/
def compLp (hg : LipschitzWith c g) (g0 : g 0 = 0) (f : Lp E p μ) : Lp F p μ :=
  ⟨AEEqFun.comp g hg.continuous (f : α →ₘ[μ] E), by
    suffices ∀ᵐ x ∂μ, ‖AEEqFun.comp g hg.continuous (f : α →ₘ[μ] E) x‖ ≤ c * ‖f x‖ from
      Lp.mem_Lp_of_ae_le_mul this
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      g : E → F
      c : NNReal
      hg : LipschitzWith c g
      g0 : Eq (g 0) 0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (↑(MeasureTheory.AEEqFun.comp g …
    -/
    filter_upwards [AEEqFun.coeFn_comp g hg.continuous (f : α →ₘ[μ] E)] with a ha
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      g : E → F
      c : NNReal
      hg : LipschitzWith c g
      g0 : Eq (g 0) 0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      a : α
      ha : Eq (↑(MeasureTheory.AEEqFun.comp g ⋯ ↑f) a) (Function.comp g (↑↑f) a)
      ⊢ LE.le (Norm.norm (↑(MeasureTheory.AEEqFun.comp g ⋯ ↑f) a)) (HMul.hMul (↑c) ( …
    -/
    simp only [ha]
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      g : E → F
      c : NNReal
      hg : LipschitzWith c g
      g0 : Eq (g 0) 0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      a : α
      ha : Eq (↑(MeasureTheory.AEEqFun.comp g ⋯ ↑f) a) (Function.comp g (↑↑f) a)
      ⊢ LE.le (Norm.norm (Function.comp g (↑↑f) a)) (HMul.hMul (↑c) (Norm.norm (↑↑f  …
    -/
    rw [← dist_zero_right, ← dist_zero_right, ← g0]
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedAddCommGroup G
      g : E → F
      c : NNReal
      hg : LipschitzWith c g
      g0 : Eq (g 0) 0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      a : α
      ha : Eq (↑(MeasureTheory.AEEqFun.comp g ⋯ ↑f) a) (Function.comp g (↑↑f) a)
      ⊢ LE.le (Dist.dist (Function.comp g (↑↑f) a) (g 0)) (HMul.hMul (↑c) (Dist.dist …
    -/
    exact hg.dist_le_mul (f a) 0⟩
    /-
      🎉 no goals
    -/


theorem coeFn_compLp (hg : LipschitzWith c g) (g0 : g 0 = 0) (f : Lp E p μ) :
    hg.compLp g0 f =ᵐ[μ] g ∘ f :=
  AEEqFun.coeFn_comp _ hg.continuous _


@[simp]
theorem compLp_zero (hg : LipschitzWith c g) (g0 : g 0 = 0) : hg.compLp g0 (0 : Lp E p μ) = 0 := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    ⊢ Eq (hg.compLp g0 0) 0
  -/
  rw [Lp.eq_zero_iff_ae_eq_zero]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(hg.compLp g0 0)) 0
  -/
  apply (coeFn_compLp _ _ _).trans
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Function.comp g ↑↑0) 0
  -/
  filter_upwards [Lp.coeFn_zero E p μ] with _ ha
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    a✝ : α
    ha : Eq (↑↑0 a✝) (0 a✝)
    ⊢ Eq (Function.comp g (↑↑0) a✝) (0 a✝)
  -/
  simp only [ha, g0, Function.comp_apply, Pi.zero_apply]
  /-
    🎉 no goals
  -/


theorem norm_compLp_sub_le (hg : LipschitzWith c g) (g0 : g 0 = 0) (f f' : Lp E p μ) :
    ‖hg.compLp g0 f - hg.compLp g0 f'‖ ≤ c * ‖f - f'‖ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    f f' : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ LE.le (Norm.norm (HSub.hSub (hg.compLp g0 f) (hg.compLp g0 f'))) (HMul.hMul  …
  -/
  apply Lp.norm_le_mul_norm_of_ae_le_mul
  filter_upwards [hg.coeFn_compLp g0 f, hg.coeFn_compLp g0 f',
    Lp.coeFn_sub (hg.compLp g0 f) (hg.compLp g0 f'), Lp.coeFn_sub f f'] with a ha1 ha2 ha3 ha4
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    f f' : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    a : α
    ha1 : Eq (↑↑(hg.compLp g0 f) a) (Function.comp g (↑↑f) a)
    ha2 : Eq (↑↑(hg.compLp g0 f') a) (Function.comp g (↑↑f') a)
    ha3 : Eq (↑↑(HSub.hSub (hg.compLp g0 f) (hg.compLp g0 f')) a) (HSub.hSub (↑↑(h …
    ha4 : Eq (↑↑(HSub.hSub f f') a) (HSub.hSub (↑↑f) (↑↑f') a)
    ⊢ LE.le (Norm.norm (↑↑(HSub.hSub (hg.compLp g0 f) (hg.compLp g0 f')) a)) (HMul …
  -/
  simp only [ha1, ha2, ha3, ha4, ← dist_eq_norm, Pi.sub_apply, Function.comp_apply]
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    g : E → F
    c : NNReal
    hg : LipschitzWith c g
    g0 : Eq (g 0) 0
    f f' : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    a : α
    ha1 : Eq (↑↑(hg.compLp g0 f) a) (Function.comp g (↑↑f) a)
    ha2 : Eq (↑↑(hg.compLp g0 f') a) (Function.comp g (↑↑f') a)
    ha3 : Eq (↑↑(HSub.hSub (hg.compLp g0 f) (hg.compLp g0 f')) a) (HSub.hSub (↑↑(h …
    ha4 : Eq (↑↑(HSub.hSub f f') a) (HSub.hSub (↑↑f) (↑↑f') a)
    ⊢ LE.le (Dist.dist (g (↑↑f a)) (g (↑↑f' a))) (HMul.hMul (↑c) (Dist.dist (↑↑f a …
  -/
  exact hg.dist_le_mul (f a) (f' a)
  /-
    🎉 no goals
  -/


theorem norm_compLp_le (hg : LipschitzWith c g) (g0 : g 0 = 0) (f : Lp E p μ) :
                                     /-
                                       α : Type u_1
                                       E : Type u_2
                                       F : Type u_3
                                       m0 : MeasurableSpace α
                                       p : ENNReal
                                       μ : MeasureTheory.Measure α
                                       inst✝¹ : NormedAddCommGroup E
                                       inst✝ : NormedAddCommGroup F
                                       g : E → F
                                       c : NNReal
                                       hg : LipschitzWith c g
                                       g0 : Eq (g 0) 0
                                       f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
                                       ⊢ LE.le (Norm.norm (hg.compLp g0 f)) (HMul.hMul (↑c) (Norm.norm f))
                                     -/
    ‖hg.compLp g0 f‖ ≤ c * ‖f‖ := by simpa using hg.norm_compLp_sub_le g0 f 0
                                     /-
                                       🎉 no goals
                                     -/


theorem lipschitzWith_compLp [Fact (1 ≤ p)] (hg : LipschitzWith c g) (g0 : g 0 = 0) :
    LipschitzWith c (hg.compLp g0 : Lp E p μ → Lp F p μ) :=
                                             /-
                                               α : Type u_1
                                               E : Type u_2
                                               F : Type u_3
                                               m0 : MeasurableSpace α
                                               p : ENNReal
                                               μ : MeasureTheory.Measure α
                                               inst✝² : NormedAddCommGroup E
                                               inst✝¹ : NormedAddCommGroup F
                                               g✝ : E → F
                                               c : NNReal
                                               inst✝ : Fact (LE.le 1 p)
                                               hg : LipschitzWith c g✝
                                               g0 : Eq (g✝ 0) 0
                                               f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
                                               ⊢ LE.le (Dist.dist (hg.compLp g0 f) (hg.compLp g0 g)) (HMul.hMul (↑c) (Dist.di …
                                             -/
  LipschitzWith.of_dist_le_mul fun f g => by simp [dist_eq_norm, norm_compLp_sub_le]
                                             /-
                                               🎉 no goals
                                             -/


theorem continuous_compLp [Fact (1 ≤ p)] (hg : LipschitzWith c g) (g0 : g 0 = 0) :
    Continuous (hg.compLp g0 : Lp E p μ → Lp F p μ) :=
  (lipschitzWith_compLp hg g0).continuous


/-- Composing `f : Lp` with `L : E →L[𝕜] F`. -/
def compLp (L : E →L[𝕜] F) (f : Lp E p μ) : Lp F p μ :=
  L.lipschitz.compLp (map_zero L) f


theorem coeFn_compLp (L : E →L[𝕜] F) (f : Lp E p μ) : ∀ᵐ a ∂μ, (L.compLp f) a = L (f a) :=
  LipschitzWith.coeFn_compLp _ _ _


theorem coeFn_compLp' (L : E →L[𝕜] F) (f : Lp E p μ) : L.compLp f =ᵐ[μ] fun a => L (f a) :=
  L.coeFn_compLp f


theorem comp_memℒp (L : E →L[𝕜] F) (f : Lp E p μ) : Memℒp (L ∘ f) p μ :=
  (Lp.memℒp (L.compLp f)).ae_eq (L.coeFn_compLp' f)


theorem comp_memℒp' (L : E →L[𝕜] F) {f : α → E} (hf : Memℒp f p μ) : Memℒp (L ∘ f) p μ :=
  (L.comp_memℒp (hf.toLp f)).ae_eq (EventuallyEq.fun_comp hf.coeFn_toLp _)


theorem _root_.MeasureTheory.Memℒp.ofReal {f : α → ℝ} (hf : Memℒp f p μ) :
    Memℒp (fun x => (f x : K)) p μ :=
  (@RCLike.ofRealCLM K _).comp_memℒp' hf


theorem _root_.MeasureTheory.memℒp_re_im_iff {f : α → K} :
    Memℒp (fun x ↦ RCLike.re (f x)) p μ ∧ Memℒp (fun x ↦ RCLike.im (f x)) p μ ↔
      Memℒp f p μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    K : Type u_6
    inst✝ : RCLike K
    f : α → K
    ⊢ Iff (And (MeasureTheory.Memℒp (fun x => RCLike.re (f x)) p μ) (MeasureTheory …
  -/
  refine ⟨?_, fun hf => ⟨hf.re, hf.im⟩⟩
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    K : Type u_6
    inst✝ : RCLike K
    f : α → K
    ⊢ And (MeasureTheory.Memℒp (fun x => RCLike.re (f x)) p μ) (MeasureTheory.Memℒ …
  -/
  rintro ⟨hre, him⟩
  /-
    case intro
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    K : Type u_6
    inst✝ : RCLike K
    f : α → K
    hre : MeasureTheory.Memℒp (fun x => RCLike.re (f x)) p μ
    him : MeasureTheory.Memℒp (fun x => RCLike.im (f x)) p μ
    ⊢ MeasureTheory.Memℒp f p μ
  -/
  convert MeasureTheory.Memℒp.add (E := K) hre.ofReal (him.ofReal.const_mul RCLike.I)
  /-
    case h.e'_6
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    K : Type u_6
    inst✝ : RCLike K
    f : α → K
    hre : MeasureTheory.Memℒp (fun x => RCLike.re (f x)) p μ
    him : MeasureTheory.Memℒp (fun x => RCLike.im (f x)) p μ
    ⊢ Eq f (HAdd.hAdd (fun x => ↑(RCLike.re (f x))) fun x => HMul.hMul RCLike.I ↑( …
  -/
  ext1 x
  /-
    case h.e'_6.h
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    K : Type u_6
    inst✝ : RCLike K
    f : α → K
    hre : MeasureTheory.Memℒp (fun x => RCLike.re (f x)) p μ
    him : MeasureTheory.Memℒp (fun x => RCLike.im (f x)) p μ
    x : α
    ⊢ Eq (f x) (HAdd.hAdd (fun x => ↑(RCLike.re (f x))) (fun x => HMul.hMul RCLike …
  -/
  rw [Pi.add_apply, mul_comm, RCLike.re_add_im]
  /-
    🎉 no goals
  -/


theorem add_compLp (L L' : E →L[𝕜] F) (f : Lp E p μ) :
    (L + L').compLp f = L.compLp f + L'.compLp f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    L L' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Eq ((HAdd.hAdd L L').compLp f) (HAdd.hAdd (L.compLp f) (L'.compLp f))
  -/
  ext1
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    L L' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((HAdd.hAdd L L').compLp f) ↑↑(HAdd.hAdd …
  -/
  refine (coeFn_compLp' (L + L') f).trans ?_
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    L L' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (HAdd.hAdd L L') (↑↑f a)) ↑↑(HAd …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_add _ _).symm
  refine
    EventuallyEq.trans ?_ (EventuallyEq.add (L.coeFn_compLp' f).symm (L'.coeFn_compLp' f).symm)
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    L L' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (HAdd.hAdd L L') (↑↑f a)) fun x  …
  -/
  filter_upwards with x
  /-
    case h.h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 F
    L L' : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    x : α
    ⊢ Eq ((HAdd.hAdd L L') (↑↑f x)) (HAdd.hAdd (L (↑↑f x)) (L' (↑↑f x)))
  -/
  rw [coe_add', Pi.add_def]
  /-
    🎉 no goals
  -/


theorem smul_compLp {𝕜'} [NormedRing 𝕜'] [Module 𝕜' F] [BoundedSMul 𝕜' F] [SMulCommClass 𝕜 𝕜' F]
    (c : 𝕜') (L : E →L[𝕜] F) (f : Lp E p μ) : (c • L).compLp f = c • L.compLp f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 F
    𝕜' : Type u_6
    inst✝³ : NormedRing 𝕜'
    inst✝² : Module 𝕜' F
    inst✝¹ : BoundedSMul 𝕜' F
    inst✝ : SMulCommClass 𝕜 𝕜' F
    c : 𝕜'
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Eq ((HSMul.hSMul c L).compLp f) (HSMul.hSMul c (L.compLp f))
  -/
  ext1
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 F
    𝕜' : Type u_6
    inst✝³ : NormedRing 𝕜'
    inst✝² : Module 𝕜' F
    inst✝¹ : BoundedSMul 𝕜' F
    inst✝ : SMulCommClass 𝕜 𝕜' F
    c : 𝕜'
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((HSMul.hSMul c L).compLp f) ↑↑(HSMul.hS …
  -/
  refine (coeFn_compLp' (c • L) f).trans ?_
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 F
    𝕜' : Type u_6
    inst✝³ : NormedRing 𝕜'
    inst✝² : Module 𝕜' F
    inst✝¹ : BoundedSMul 𝕜' F
    inst✝ : SMulCommClass 𝕜 𝕜' F
    c : 𝕜'
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (HSMul.hSMul c L) (↑↑f a)) ↑↑(HS …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_smul _ _).symm
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 F
    𝕜' : Type u_6
    inst✝³ : NormedRing 𝕜'
    inst✝² : Module 𝕜' F
    inst✝¹ : BoundedSMul 𝕜' F
    inst✝ : SMulCommClass 𝕜 𝕜' F
    c : 𝕜'
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => (HSMul.hSMul c L) (↑↑f a)) (HSMu …
  -/
  refine (L.coeFn_compLp' f).mono fun x hx => ?_
  /-
    case h
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜 F
    𝕜' : Type u_6
    inst✝³ : NormedRing 𝕜'
    inst✝² : Module 𝕜' F
    inst✝¹ : BoundedSMul 𝕜' F
    inst✝ : SMulCommClass 𝕜 𝕜' F
    c : 𝕜'
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    x : α
    hx : Eq (↑↑(L.compLp f) x) ((fun a => L (↑↑f a)) x)
    ⊢ Eq ((fun a => (HSMul.hSMul c L) (↑↑f a)) x) (HSMul.hSMul c (↑↑(L.compLp f)) x)
  -/
  rw [Pi.smul_apply, hx, coe_smul', Pi.smul_def]
  /-
    🎉 no goals
  -/


theorem norm_compLp_le (L : E →L[𝕜] F) (f : Lp E p μ) : ‖L.compLp f‖ ≤ ‖L‖ * ‖f‖ :=
  LipschitzWith.norm_compLp_le _ _ _


/-- Composing `f : Lp E p μ` with `L : E →L[𝕜] F`, seen as a `𝕜`-linear map on `Lp E p μ`. -/
def compLpₗ (L : E →L[𝕜] F) : Lp E p μ →ₗ[𝕜] Lp F p μ where
  toFun f := L.compLp f
  map_add' f g := by
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedAddCommGroup G
      g✝ : E → F
      c : NNReal
      𝕜 : Type u_5
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ Eq ((fun f => L.compLp f) (HAdd.hAdd f g)) (HAdd.hAdd ((fun f => L.compLp f) …
    -/
    ext1
    filter_upwards [Lp.coeFn_add f g, coeFn_compLp L (f + g), coeFn_compLp L f,
      coeFn_compLp L g, Lp.coeFn_add (L.compLp f) (L.compLp g)]
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedAddCommGroup G
      g✝ : E → F
      c : NNReal
      𝕜 : Type u_5
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ ∀ (a : α), Eq (↑↑(HAdd.hAdd f g) a) (HAdd.hAdd (↑↑f) (↑↑g) a) → Eq (↑↑(L.com …
    -/
    intro a ha1 ha2 ha3 ha4 ha5
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedAddCommGroup G
      g✝ : E → F
      c : NNReal
      𝕜 : Type u_5
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      a : α
      ha1 : Eq (↑↑(HAdd.hAdd f g) a) (HAdd.hAdd (↑↑f) (↑↑g) a)
      ha2 : Eq (↑↑(L.compLp (HAdd.hAdd f g)) a) (L (↑↑(HAdd.hAdd f g) a))
      ha3 : Eq (↑↑(L.compLp f) a) (L (↑↑f a))
      ha4 : Eq (↑↑(L.compLp g) a) (L (↑↑g a))
      ha5 : Eq (↑↑(HAdd.hAdd (L.compLp f) (L.compLp g)) a) (HAdd.hAdd (↑↑(L.compLp f …
      ⊢ Eq (↑↑(L.compLp (HAdd.hAdd f g)) a) (↑↑(HAdd.hAdd (L.compLp f) (L.compLp g)) …
    -/
    simp only [ha1, ha2, ha3, ha4, ha5, map_add, Pi.add_apply]
    /-
      🎉 no goals
    -/
  map_smul' c f := by
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedAddCommGroup G
      g : E → F
      c✝ : NNReal
      𝕜 : Type u_5
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      c : 𝕜
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ Eq ({ toFun := fun f => L.compLp f, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) …
    -/
    dsimp
    /-
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedAddCommGroup G
      g : E → F
      c✝ : NNReal
      𝕜 : Type u_5
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      c : 𝕜
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      ⊢ Eq (L.compLp (HSMul.hSMul c f)) (HSMul.hSMul c (L.compLp f))
    -/
    ext1
    filter_upwards [Lp.coeFn_smul c f, coeFn_compLp L (c • f), Lp.coeFn_smul c (L.compLp f),
      coeFn_compLp L f] with _ ha1 ha2 ha3 ha4
    /-
      case h
      α : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      m m0 : MeasurableSpace α
      p : ENNReal
      q : Real
      μ ν : MeasureTheory.Measure α
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedAddCommGroup G
      g : E → F
      c✝ : NNReal
      𝕜 : Type u_5
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 F
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      c : 𝕜
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      a✝ : α
      ha1 : Eq (↑↑(HSMul.hSMul c f) a✝) (HSMul.hSMul c (↑↑f) a✝)
      ha2 : Eq (↑↑(L.compLp (HSMul.hSMul c f)) a✝) (L (↑↑(HSMul.hSMul c f) a✝))
      ha3 : Eq (↑↑(HSMul.hSMul c (L.compLp f)) a✝) (HSMul.hSMul c (↑↑(L.compLp f)) a✝)
      ha4 : Eq (↑↑(L.compLp f) a✝) (L (↑↑f a✝))
      ⊢ Eq (↑↑(L.compLp (HSMul.hSMul c f)) a✝) (↑↑(HSMul.hSMul c (L.compLp f)) a✝)
    -/
    simp only [ha1, ha2, ha3, ha4, _root_.map_smul, Pi.smul_apply]
    /-
      🎉 no goals
    -/


/-- Composing `f : Lp E p μ` with `L : E →L[𝕜] F`, seen as a continuous `𝕜`-linear map on
`Lp E p μ`. See also the similar
* `LinearMap.compLeft` for functions,
* `ContinuousLinearMap.compLeftContinuous` for continuous functions,
* `ContinuousLinearMap.compLeftContinuousBounded` for bounded continuous functions,
* `ContinuousLinearMap.compLeftContinuousCompact` for continuous functions on compact spaces.
-/
def compLpL [Fact (1 ≤ p)] (L : E →L[𝕜] F) : Lp E p μ →L[𝕜] Lp F p μ :=
  LinearMap.mkContinuous (L.compLpₗ p μ) ‖L‖ L.norm_compLp_le


theorem coeFn_compLpL [Fact (1 ≤ p)] (L : E →L[𝕜] F) (f : Lp E p μ) :
    L.compLpL p μ f =ᵐ[μ] fun a => L (f a) :=
  L.coeFn_compLp f


theorem add_compLpL [Fact (1 ≤ p)] (L L' : E →L[𝕜] F) :
                                                                /-
                                                                  α : Type u_1
                                                                  E : Type u_2
                                                                  F : Type u_3
                                                                  m0 : MeasurableSpace α
                                                                  p : ENNReal
                                                                  μ : MeasureTheory.Measure α
                                                                  inst✝⁵ : NormedAddCommGroup E
                                                                  inst✝⁴ : NormedAddCommGroup F
                                                                  𝕜 : Type u_5
                                                                  inst✝³ : NontriviallyNormedField 𝕜
                                                                  inst✝² : NormedSpace 𝕜 E
                                                                  inst✝¹ : NormedSpace 𝕜 F
                                                                  inst✝ : Fact (LE.le 1 p)
                                                                  L L' : ContinuousLinearMap (RingHom.id 𝕜) E F
                                                                  ⊢ Eq (ContinuousLinearMap.compLpL p μ (HAdd.hAdd L L')) (HAdd.hAdd (Continuous …
                                                                -/
    (L + L').compLpL p μ = L.compLpL p μ + L'.compLpL p μ := by ext1 f; exact add_compLp L L' f
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem smul_compLpL [Fact (1 ≤ p)] {𝕜'} [NormedRing 𝕜'] [Module 𝕜' F] [BoundedSMul 𝕜' F]
    [SMulCommClass 𝕜 𝕜' F] (c : 𝕜') (L : E →L[𝕜] F) : (c • L).compLpL p μ = c • L.compLpL p μ := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup F
    𝕜 : Type u_5
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜 F
    inst✝⁴ : Fact (LE.le 1 p)
    𝕜' : Type u_6
    inst✝³ : NormedRing 𝕜'
    inst✝² : Module 𝕜' F
    inst✝¹ : BoundedSMul 𝕜' F
    inst✝ : SMulCommClass 𝕜 𝕜' F
    c : 𝕜'
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (ContinuousLinearMap.compLpL p μ (HSMul.hSMul c L)) (HSMul.hSMul c (Conti …
  -/
  ext1 f; exact smul_compLp c L f
          /-
            🎉 no goals
          -/


theorem norm_compLpL_le [Fact (1 ≤ p)] (L : E →L[𝕜] F) : ‖L.compLpL p μ‖ ≤ ‖L‖ :=
  LinearMap.mkContinuous_norm_le _ (norm_nonneg _) _


theorem indicatorConstLp_eq_toSpanSingleton_compLp {s : Set α} [NormedSpace ℝ F]
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : F) :
    indicatorConstLp 2 hs hμs x =
      (ContinuousLinearMap.toSpanSingleton ℝ x).compLp (indicatorConstLp 2 hs hμs (1 : ℝ)) := by
  /-
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    ⊢ Eq (MeasureTheory.indicatorConstLp 2 hs hμs x) ((ContinuousLinearMap.toSpanS …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.indicatorConstLp 2 hs hμs …
  -/
  refine indicatorConstLp_coeFn.trans ?_
  have h_compLp :=
    (ContinuousLinearMap.toSpanSingleton ℝ x).coeFn_compLp (indicatorConstLp 2 hs hμs (1 : ℝ))
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : Filter.Eventually (fun a => Eq (↑↑((ContinuousLinearMap.toSpanSingl …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator fun x_1 => x) ↑↑((ContinuousL …
  -/
  rw [← EventuallyEq] at h_compLp
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSin …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator fun x_1 => x) ↑↑((ContinuousL …
  -/
  refine EventuallyEq.trans ?_ h_compLp.symm
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSin …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator fun x_1 => x) fun a => (Conti …
  -/
  refine (@indicatorConstLp_coeFn _ _ _ 2 μ _ s hs hμs (1 : ℝ)).mono fun y hy => ?_
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSin …
    y : α
    hy : Eq (↑↑(MeasureTheory.indicatorConstLp 2 hs hμs 1) y) (s.indicator (fun x  …
    ⊢ Eq (s.indicator (fun x_1 => x) y) ((fun a => (ContinuousLinearMap.toSpanSing …
  -/
  dsimp only
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSin …
    y : α
    hy : Eq (↑↑(MeasureTheory.indicatorConstLp 2 hs hμs 1) y) (s.indicator (fun x  …
    ⊢ Eq (s.indicator (fun x_1 => x) y) ((ContinuousLinearMap.toSpanSingleton Real …
  -/
  rw [hy]
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSin …
    y : α
    hy : Eq (↑↑(MeasureTheory.indicatorConstLp 2 hs hμs 1) y) (s.indicator (fun x  …
    ⊢ Eq (s.indicator (fun x_1 => x) y) ((ContinuousLinearMap.toSpanSingleton Real …
  -/
  simp_rw [ContinuousLinearMap.toSpanSingleton_apply]
  /-
    case h
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup F
    s : Set α
    inst✝ : NormedSpace Real F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F
    h_compLp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSin …
    y : α
    hy : Eq (↑↑(MeasureTheory.indicatorConstLp 2 hs hμs 1) y) (s.indicator (fun x  …
    ⊢ Eq (s.indicator (fun x_1 => x) y) (HSMul.hSMul (s.indicator (fun x => 1) y) x)
  -/
                              /-
                                🎉 no goals
                              -/
  by_cases hy_mem : y ∈ s <;> simp [hy_mem]
                              /-
                                🎉 no goals
                              -/


theorem lipschitzWith_pos_part : LipschitzWith 1 fun x : ℝ => max x 0 :=
  LipschitzWith.id.max_const _


theorem _root_.MeasureTheory.Memℒp.pos_part {f : α → ℝ} (hf : Memℒp f p μ) :
    Memℒp (fun x => max (f x) 0) p μ :=
  lipschitzWith_pos_part.comp_memℒp (max_eq_right le_rfl) hf


theorem _root_.MeasureTheory.Memℒp.neg_part {f : α → ℝ} (hf : Memℒp f p μ) :
    Memℒp (fun x => max (-f x) 0) p μ :=
  lipschitzWith_pos_part.comp_memℒp (max_eq_right le_rfl) hf.neg


/-- Positive part of a function in `L^p`. -/
def posPart (f : Lp ℝ p μ) : Lp ℝ p μ :=
  lipschitzWith_pos_part.compLp (max_eq_right le_rfl) f


/-- Negative part of a function in `L^p`. -/
def negPart (f : Lp ℝ p μ) : Lp ℝ p μ :=
  posPart (-f)


@[norm_cast]
theorem coe_posPart (f : Lp ℝ p μ) : (posPart f : α →ₘ[μ] ℝ) = (f : α →ₘ[μ] ℝ).posPart :=
  rfl


theorem coeFn_posPart (f : Lp ℝ p μ) : ⇑(posPart f) =ᵐ[μ] fun a => max (f a) 0 :=
  AEEqFun.coeFn_posPart _


theorem coeFn_negPart_eq_max (f : Lp ℝ p μ) : ∀ᵐ a ∂μ, negPart f a = max (-f a) 0 := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real p μ) x
    ⊢ Filter.Eventually (fun a => Eq (↑↑(MeasureTheory.Lp.negPart f) a) (Max.max ( …
  -/
  rw [negPart]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real p μ) x
    ⊢ Filter.Eventually (fun a => Eq (↑↑(MeasureTheory.Lp.posPart (Neg.neg f)) a)  …
  -/
  filter_upwards [coeFn_posPart (-f), coeFn_neg f] with _ h₁ h₂
  /-
    case h
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real p μ) x
    a✝ : α
    h₁ : Eq (↑↑(MeasureTheory.Lp.posPart (Neg.neg f)) a✝) (Max.max (↑↑(Neg.neg f)  …
    h₂ : Eq (↑↑(Neg.neg f) a✝) (Neg.neg (↑↑f) a✝)
    ⊢ Eq (↑↑(MeasureTheory.Lp.posPart (Neg.neg f)) a✝) (Max.max (Neg.neg (↑↑f a✝)) …
  -/
  rw [h₁, h₂, Pi.neg_apply]
  /-
    🎉 no goals
  -/


theorem coeFn_negPart (f : Lp ℝ p μ) : ∀ᵐ a ∂μ, negPart f a = -min (f a) 0 :=
                                              /-
                                                α : Type u_1
                                                m0 : MeasurableSpace α
                                                p : ENNReal
                                                μ : MeasureTheory.Measure α
                                                f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real p μ) x
                                                a : α
                                                h : Eq (↑↑(MeasureTheory.Lp.negPart f) a) (Max.max (Neg.neg (↑↑f a)) 0)
                                                ⊢ Eq (↑↑(MeasureTheory.Lp.negPart f) a) (Neg.neg (Min.min (↑↑f a) 0))
                                              -/
  (coeFn_negPart_eq_max f).mono fun a h => by rw [h, ← max_neg_neg, neg_zero]
                                              /-
                                                🎉 no goals
                                              -/


theorem continuous_posPart [Fact (1 ≤ p)] : Continuous fun f : Lp ℝ p μ => posPart f :=
  LipschitzWith.continuous_compLp _ _


theorem continuous_negPart [Fact (1 ≤ p)] : Continuous fun f : Lp ℝ p μ => negPart f := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    ⊢ Continuous fun f => MeasureTheory.Lp.negPart f
  -/
  unfold negPart
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : Fact (LE.le 1 p)
    ⊢ Continuous fun f => MeasureTheory.Lp.posPart (Neg.neg f)
  -/
  exact continuous_posPart.comp continuous_neg
  /-
    🎉 no goals
  -/


theorem eLpNorm'_lim_eq_lintegral_liminf {ι} [Nonempty ι] [LinearOrder ι] {f : ι → α → G} {p : ℝ}
    (hp_nonneg : 0 ≤ p) {f_lim : α → G}
    (h_lim : ∀ᵐ x : α ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x))) :
    eLpNorm' f_lim p μ = (∫⁻ a, atTop.liminf fun m => (‖f m a‖₊ : ℝ≥0∞) ^ p ∂μ) ^ (1 / p) := by
  suffices h_no_pow :
      (∫⁻ a, (‖f_lim a‖₊ : ℝ≥0∞) ^ p ∂μ) = ∫⁻ a, atTop.liminf fun m => (‖f m a‖₊ : ℝ≥0∞) ^ p ∂μ by
    rw [eLpNorm'_eq_lintegral_nnnorm, h_no_pow]
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    p : Real
    hp_nonneg : LE.le 0 p
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (f_lim a)) …
  -/
  refine lintegral_congr_ae (h_lim.mono fun a ha => ?_)
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    p : Real
    hp_nonneg : LE.le 0 p
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    a : α
    ha : Filter.Tendsto (fun n => f n a) Filter.atTop (nhds (f_lim a))
    ⊢ Eq ((fun a => HPow.hPow (↑(NNNorm.nnnorm (f_lim a))) p) a) ((fun a => Filter …
  -/
  dsimp only
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    p : Real
    hp_nonneg : LE.le 0 p
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    a : α
    ha : Filter.Tendsto (fun n => f n a) Filter.atTop (nhds (f_lim a))
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm (f_lim a))) p) (Filter.liminf (fun m => HPow. …
  -/
  rw [Tendsto.liminf_eq]
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    p : Real
    hp_nonneg : LE.le 0 p
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    a : α
    ha : Filter.Tendsto (fun n => f n a) Filter.atTop (nhds (f_lim a))
    ⊢ Filter.Tendsto (fun m => HPow.hPow (↑(NNNorm.nnnorm (f m a))) p) Filter.atTo …
  -/
  simp_rw [← ENNReal.coe_rpow_of_nonneg _ hp_nonneg, ENNReal.tendsto_coe]
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    p : Real
    hp_nonneg : LE.le 0 p
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    a : α
    ha : Filter.Tendsto (fun n => f n a) Filter.atTop (nhds (f_lim a))
    ⊢ Filter.Tendsto (fun a_1 => HPow.hPow (NNNorm.nnnorm (f a_1 a)) p) Filter.atT …
  -/
  refine ((NNReal.continuous_rpow_const hp_nonneg).tendsto ‖f_lim a‖₊).comp ?_
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    p : Real
    hp_nonneg : LE.le 0 p
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    a : α
    ha : Filter.Tendsto (fun n => f n a) Filter.atTop (nhds (f_lim a))
    ⊢ Filter.Tendsto (fun a_1 => NNNorm.nnnorm (f a_1 a)) Filter.atTop (nhds (NNNo …
  -/
  exact (continuous_nnnorm.tendsto (f_lim a)).comp ha
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_lim_eq_lintegral_liminf := eLpNorm'_lim_eq_lintegral_liminf


theorem eLpNorm'_lim_le_liminf_eLpNorm' {E} [NormedAddCommGroup E] {f : ℕ → α → E} {p : ℝ}
    (hp_pos : 0 < p) (hf : ∀ n, AEStronglyMeasurable (f n) μ) {f_lim : α → E}
    (h_lim : ∀ᵐ x : α ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x))) :
    eLpNorm' f_lim p μ ≤ atTop.liminf fun n => eLpNorm' (f n) p μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp_pos : LT.lt 0 p
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (MeasureTheory.eLpNorm' f_lim p μ) (Filter.liminf (fun n => MeasureThe …
  -/
  rw [eLpNorm'_lim_eq_lintegral_liminf hp_pos.le h_lim]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp_pos : LT.lt 0 p
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => Filter.liminf (fun m => …
  -/
  rw [one_div, ← ENNReal.le_rpow_inv_iff (by simp [hp_pos] : 0 < p⁻¹), inv_inv]
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp_pos : LT.lt 0 p
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => Filter.liminf (fun m => HPow.hPow  …
  -/
  refine (lintegral_liminf_le' fun m => (hf m).ennnorm.pow_const _).trans_eq ?_
  have h_pow_liminf :
    atTop.liminf (fun n ↦ eLpNorm' (f n) p μ) ^ p
      = atTop.liminf fun n ↦ eLpNorm' (f n) p μ ^ p := by
    have h_rpow_mono := ENNReal.strictMono_rpow_of_pos hp_pos
    have h_rpow_surj := (ENNReal.rpow_left_bijective hp_pos.ne.symm).2
    refine (h_rpow_mono.orderIsoOfSurjective _ h_rpow_surj).liminf_apply ?_ ?_ ?_ ?_
    all_goals isBoundedDefault
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp_pos : LT.lt 0 p
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    h_pow_liminf : Eq (HPow.hPow (Filter.liminf (fun n => MeasureTheory.eLpNorm' ( …
    ⊢ Eq (Filter.liminf (fun n => MeasureTheory.lintegral μ fun a => HPow.hPow (↑( …
  -/
  rw [h_pow_liminf]
  simp_rw [eLpNorm'_eq_lintegral_nnnorm, ← ENNReal.rpow_mul, one_div,
    inv_mul_cancel₀ hp_pos.ne.symm, ENNReal.rpow_one]


@[deprecated (since := "2024-07-27")]
alias snorm'_lim_le_liminf_snorm' := eLpNorm'_lim_le_liminf_eLpNorm'


theorem eLpNorm_exponent_top_lim_eq_essSup_liminf {ι} [Nonempty ι] [LinearOrder ι] {f : ι → α → G}
    {f_lim : α → G} (h_lim : ∀ᵐ x : α ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x))) :
    eLpNorm f_lim ∞ μ = essSup (fun x => atTop.liminf fun m => (‖f m x‖₊ : ℝ≥0∞)) μ := by
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ Eq (MeasureTheory.eLpNorm f_lim Top.top μ) (essSup (fun x => Filter.liminf ( …
  -/
  rw [eLpNorm_exponent_top, eLpNormEssSup_eq_essSup_nnnorm]
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ Eq (essSup (fun x => ↑(NNNorm.nnnorm (f_lim x))) μ) (essSup (fun x => Filter …
  -/
  refine essSup_congr_ae (h_lim.mono fun x hx => ?_)
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    x : α
    hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (f_lim x))
    ⊢ Eq ((fun x => ↑(NNNorm.nnnorm (f_lim x))) x) ((fun x => Filter.liminf (fun m …
  -/
  dsimp only
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    x : α
    hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (f_lim x))
    ⊢ Eq (↑(NNNorm.nnnorm (f_lim x))) (Filter.liminf (fun m => ↑(NNNorm.nnnorm (f  …
  -/
  apply (Tendsto.liminf_eq ..).symm
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    x : α
    hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (f_lim x))
    ⊢ Filter.Tendsto (fun m => ↑(NNNorm.nnnorm (f m x))) Filter.atTop (nhds ↑(NNNo …
  -/
  rw [ENNReal.tendsto_coe]
  /-
    α : Type u_1
    G : Type u_4
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup G
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : LinearOrder ι
    f : ι → α → G
    f_lim : α → G
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    x : α
    hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (f_lim x))
    ⊢ Filter.Tendsto (fun m => NNNorm.nnnorm (f m x)) Filter.atTop (nhds (NNNorm.n …
  -/
  exact (continuous_nnnorm.tendsto (f_lim x)).comp hx
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_exponent_top_lim_eq_essSup_liminf := eLpNorm_exponent_top_lim_eq_essSup_liminf


theorem eLpNorm_exponent_top_lim_le_liminf_eLpNorm_exponent_top {ι} [Nonempty ι] [Countable ι]
    [LinearOrder ι] {f : ι → α → F} {f_lim : α → F}
    (h_lim : ∀ᵐ x : α ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x))) :
    eLpNorm f_lim ∞ μ ≤ atTop.liminf fun n => eLpNorm (f n) ∞ μ := by
  /-
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    ι : Type u_5
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    inst✝ : LinearOrder ι
    f : ι → α → F
    f_lim : α → F
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (MeasureTheory.eLpNorm f_lim Top.top μ) (Filter.liminf (fun n => Measu …
  -/
  rw [eLpNorm_exponent_top_lim_eq_essSup_liminf h_lim]
  /-
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    ι : Type u_5
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    inst✝ : LinearOrder ι
    f : ι → α → F
    f_lim : α → F
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (essSup (fun x => Filter.liminf (fun m => ↑(NNNorm.nnnorm (f m x))) Fi …
  -/
  simp_rw [eLpNorm_exponent_top, eLpNormEssSup]
  /-
    α : Type u_1
    F : Type u_3
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup F
    ι : Type u_5
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    inst✝ : LinearOrder ι
    f : ι → α → F
    f_lim : α → F
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (essSup (fun x => Filter.liminf (fun m => ↑(NNNorm.nnnorm (f m x))) Fi …
  -/
  exact ENNReal.essSup_liminf_le fun n => fun x => (‖f n x‖₊ : ℝ≥0∞)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_exponent_top_lim_le_liminf_snorm_exponent_top :=
  eLpNorm_exponent_top_lim_le_liminf_eLpNorm_exponent_top


theorem eLpNorm_lim_le_liminf_eLpNorm {E} [NormedAddCommGroup E] {f : ℕ → α → E}
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (f_lim : α → E)
    (h_lim : ∀ᵐ x : α ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x))) :
    eLpNorm f_lim p μ ≤ atTop.liminf fun n => eLpNorm (f n) p μ := by
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ LE.le (MeasureTheory.eLpNorm f_lim p μ) (Filter.liminf (fun n => MeasureTheo …
  -/
  obtain rfl|hp0 := eq_or_ne p 0
    /-
      case inl
      α : Type u_1
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝ : NormedAddCommGroup E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      f_lim : α → E
      h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
      ⊢ LE.le (MeasureTheory.eLpNorm f_lim 0 μ) (Filter.liminf (fun n => MeasureTheo …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    hp0 : Ne p 0
    ⊢ LE.le (MeasureTheory.eLpNorm f_lim p μ) (Filter.liminf (fun n => MeasureTheo …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝ : NormedAddCommGroup E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      f_lim : α → E
      h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
      hp0 : Ne p 0
      hp_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f_lim p μ) (Filter.liminf (fun n => MeasureTheo …
    -/
  · simp_rw [hp_top]
    /-
      case pos
      α : Type u_1
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      E : Type u_5
      inst✝ : NormedAddCommGroup E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      f_lim : α → E
      h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
      hp0 : Ne p 0
      hp_top : Eq p Top.top
      ⊢ LE.le (MeasureTheory.eLpNorm f_lim Top.top μ) (Filter.liminf (fun n => Measu …
    -/
    exact eLpNorm_exponent_top_lim_le_liminf_eLpNorm_exponent_top h_lim
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm f_lim p μ) (Filter.liminf (fun n => MeasureTheo …
  -/
  simp_rw [eLpNorm_eq_eLpNorm' hp0 hp_top]
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    ⊢ LE.le (MeasureTheory.eLpNorm' f_lim p.toReal μ) (Filter.liminf (fun n => Mea …
  -/
  have hp_pos : 0 < p.toReal := ENNReal.toReal_pos hp0 hp_top
  /-
    case neg
    α : Type u_1
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    hp0 : Ne p 0
    hp_top : Not (Eq p Top.top)
    hp_pos : LT.lt 0 p.toReal
    ⊢ LE.le (MeasureTheory.eLpNorm' f_lim p.toReal μ) (Filter.liminf (fun n => Mea …
  -/
  exact eLpNorm'_lim_le_liminf_eLpNorm' hp_pos hf h_lim
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_lim_le_liminf_snorm := eLpNorm_lim_le_liminf_eLpNorm


theorem tendsto_Lp_iff_tendsto_ℒp' {ι} {fi : Filter ι} [Fact (1 ≤ p)] (f : ι → Lp E p μ)
    (f_lim : Lp E p μ) :
    fi.Tendsto f (𝓝 f_lim) ↔ fi.Tendsto (fun n => eLpNorm (⇑(f n) - ⇑f_lim) p μ) (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff (Filter.Tendsto f fi (nhds f_lim)) (Filter.Tendsto (fun n => MeasureTheo …
  -/
  rw [tendsto_iff_dist_tendsto_zero]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff (Filter.Tendsto (fun b => Dist.dist (f b) f_lim) fi (nhds 0)) (Filter.Te …
  -/
  simp_rw [dist_def]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff (Filter.Tendsto (fun b => (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f b) ↑↑f_ …
  -/
  rw [← ENNReal.zero_toReal, ENNReal.tendsto_toReal_iff (fun n => ?_) ENNReal.zero_ne_top]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    n : ι
    ⊢ Ne (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑f_lim) p μ) Top.top
  -/
  rw [eLpNorm_congr_ae (Lp.coeFn_sub _ _).symm]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    n : ι
    ⊢ Ne (MeasureTheory.eLpNorm (↑↑(HSub.hSub (f n) f_lim)) p μ) Top.top
  -/
  exact Lp.eLpNorm_ne_top _
  /-
    🎉 no goals
  -/


theorem tendsto_Lp_iff_tendsto_ℒp {ι} {fi : Filter ι} [Fact (1 ≤ p)] (f : ι → Lp E p μ)
    (f_lim : α → E) (f_lim_ℒp : Memℒp f_lim p μ) :
    fi.Tendsto f (𝓝 (f_lim_ℒp.toLp f_lim)) ↔
      fi.Tendsto (fun n => eLpNorm (⇑(f n) - f_lim) p μ) (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    ⊢ Iff (Filter.Tendsto f fi (nhds (MeasureTheory.Memℒp.toLp f_lim f_lim_ℒp))) ( …
  -/
  rw [tendsto_Lp_iff_tendsto_ℒp']
  suffices h_eq :
      (fun n => eLpNorm (⇑(f n) - ⇑(Memℒp.toLp f_lim f_lim_ℒp)) p μ) =
        (fun n => eLpNorm (⇑(f n) - f_lim) p μ) by
    rw [h_eq]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    ⊢ Eq (fun n => MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(MeasureTheory.Memℒp …
  -/
  exact funext fun n => eLpNorm_congr_ae (EventuallyEq.rfl.sub (Memℒp.coeFn_toLp f_lim_ℒp))
  /-
    🎉 no goals
  -/


theorem tendsto_Lp_iff_tendsto_ℒp'' {ι} {fi : Filter ι} [Fact (1 ≤ p)] (f : ι → α → E)
    (f_ℒp : ∀ n, Memℒp (f n) p μ) (f_lim : α → E) (f_lim_ℒp : Memℒp f_lim p μ) :
    fi.Tendsto (fun n => (f_ℒp n).toLp (f n)) (𝓝 (f_lim_ℒp.toLp f_lim)) ↔
      fi.Tendsto (fun n => eLpNorm (f n - f_lim) p μ) (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → α → E
    f_ℒp : ∀ (n : ι), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    ⊢ Iff (Filter.Tendsto (fun n => MeasureTheory.Memℒp.toLp (f n) ⋯) fi (nhds (Me …
  -/
  rw [Lp.tendsto_Lp_iff_tendsto_ℒp' (fun n => (f_ℒp n).toLp (f n)) (f_lim_ℒp.toLp f_lim)]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → α → E
    f_ℒp : ∀ (n : ι), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    ⊢ Iff (Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub ↑↑(MeasureThe …
  -/
  refine Filter.tendsto_congr fun n => ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → α → E
    f_ℒp : ∀ (n : ι), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    n : ι
    ⊢ Eq (MeasureTheory.eLpNorm (HSub.hSub ↑↑(MeasureTheory.Memℒp.toLp (f n) ⋯) ↑↑ …
  -/
  apply eLpNorm_congr_ae
  filter_upwards [((f_ℒp n).sub f_lim_ℒp).coeFn_toLp,
    Lp.coeFn_sub ((f_ℒp n).toLp (f n)) (f_lim_ℒp.toLp f_lim)] with _ hx₁ hx₂
  /-
    case h
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → α → E
    f_ℒp : ∀ (n : ι), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    n : ι
    a✝ : α
    hx₁ : Eq (↑↑(MeasureTheory.Memℒp.toLp (HSub.hSub (f n) f_lim) ⋯) a✝) (HSub.hSu …
    hx₂ : Eq (↑↑(HSub.hSub (MeasureTheory.Memℒp.toLp (f n) ⋯) (MeasureTheory.Memℒp …
    ⊢ Eq (HSub.hSub (↑↑(MeasureTheory.Memℒp.toLp (f n) ⋯)) (↑↑(MeasureTheory.Memℒp …
  -/
  rw [← hx₂]
  /-
    case h
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    ι : Type u_5
    fi : Filter ι
    inst✝ : Fact (LE.le 1 p)
    f : ι → α → E
    f_ℒp : ∀ (n : ι), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    f_lim_ℒp : MeasureTheory.Memℒp f_lim p μ
    n : ι
    a✝ : α
    hx₁ : Eq (↑↑(MeasureTheory.Memℒp.toLp (HSub.hSub (f n) f_lim) ⋯) a✝) (HSub.hSu …
    hx₂ : Eq (↑↑(HSub.hSub (MeasureTheory.Memℒp.toLp (f n) ⋯) (MeasureTheory.Memℒp …
    ⊢ Eq (↑↑(HSub.hSub (MeasureTheory.Memℒp.toLp (f n) ⋯) (MeasureTheory.Memℒp.toL …
  -/
  exact hx₁
  /-
    🎉 no goals
  -/


theorem tendsto_Lp_of_tendsto_ℒp {ι} {fi : Filter ι} [Fact (1 ≤ p)] {f : ι → Lp E p μ}
    (f_lim : α → E) (f_lim_ℒp : Memℒp f_lim p μ)
    (h_tendsto : fi.Tendsto (fun n => eLpNorm (⇑(f n) - f_lim) p μ) (𝓝 0)) :
    fi.Tendsto f (𝓝 (f_lim_ℒp.toLp f_lim)) :=
  (tendsto_Lp_iff_tendsto_ℒp f f_lim f_lim_ℒp).mpr h_tendsto


theorem cauchySeq_Lp_iff_cauchySeq_ℒp {ι} [Nonempty ι] [SemilatticeSup ι] [hp : Fact (1 ≤ p)]
    (f : ι → Lp E p μ) :
    CauchySeq f ↔ Tendsto (fun n : ι × ι => eLpNorm (⇑(f n.fst) - ⇑(f n.snd)) p μ) atTop (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : SemilatticeSup ι
    hp : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff (CauchySeq f) (Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub …
  -/
  simp_rw [cauchySeq_iff_tendsto_dist_atTop_0, dist_def]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : SemilatticeSup ι
    hp : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff (Filter.Tendsto (fun n => (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n.1) ↑↑ …
  -/
  rw [← ENNReal.zero_toReal, ENNReal.tendsto_toReal_iff (fun n => ?_) ENNReal.zero_ne_top]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : SemilatticeSup ι
    hp : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    n : Prod ι ι
    ⊢ Ne (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n.1) ↑↑(f n.2)) p μ) Top.top
  -/
  rw [eLpNorm_congr_ae (Lp.coeFn_sub _ _).symm]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup E
    ι : Type u_5
    inst✝¹ : Nonempty ι
    inst✝ : SemilatticeSup ι
    hp : Fact (LE.le 1 p)
    f : ι → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    n : Prod ι ι
    ⊢ Ne (MeasureTheory.eLpNorm (↑↑(HSub.hSub (f n.1) (f n.2))) p μ) Top.top
  -/
  exact eLpNorm_ne_top _
  /-
    🎉 no goals
  -/


theorem completeSpace_lp_of_cauchy_complete_ℒp [hp : Fact (1 ≤ p)]
    (H :
      ∀ (f : ℕ → α → E) (_ : ∀ n, Memℒp (f n) p μ) (B : ℕ → ℝ≥0∞) (_ : ∑' i, B i < ∞)
        (_ : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm (f n - f m) p μ < B N),
        ∃ (f_lim : α → E), Memℒp f_lim p μ ∧
          atTop.Tendsto (fun n => eLpNorm (f n - f_lim) p μ) (𝓝 0)) :
    CompleteSpace (Lp E p μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    ⊢ CompleteSpace (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x)
  -/
  let B := fun n : ℕ => ((1 : ℝ) / 2) ^ n
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    ⊢ CompleteSpace (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x)
  -/
  have hB_pos : ∀ n, 0 < B n := fun n => pow_pos (div_pos zero_lt_one zero_lt_two) n
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    ⊢ CompleteSpace (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x)
  -/
  refine Metric.complete_of_convergent_controlled_sequences B hB_pos fun f hf => ?_
  rsuffices ⟨f_lim, hf_lim_meas, h_tendsto⟩ :
    ∃ (f_lim : α → E), Memℒp f_lim p μ ∧
      atTop.Tendsto (fun n => eLpNorm (⇑(f n) - f_lim) p μ) (𝓝 0)
    /-
      case intro.intro
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hp : Fact (LE.le 1 p)
      H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
      B : Nat → Real := fun n => HPow.hPow (1 / 2) n
      hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
      f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      hf : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (f n) (f m)) (B …
      f_lim : α → E
      hf_lim_meas : MeasureTheory.Memℒp f_lim p μ
      h_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (↑↑(f n) …
      ⊢ Exists fun x => Filter.Tendsto f Filter.atTop (nhds x)
    -/
  · exact ⟨hf_lim_meas.toLp f_lim, tendsto_Lp_of_tendsto_ℒp f_lim hf_lim_meas h_tendsto⟩
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hf : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (f n) (f m)) (B …
    ⊢ Exists fun f_lim => And (MeasureTheory.Memℒp f_lim p μ) (Filter.Tendsto (fun …
  -/
  obtain ⟨M, hB⟩ : Summable B := summable_geometric_two
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hf : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (f n) (f m)) (B …
    M : Real
    hB : HasSum B M
    ⊢ Exists fun f_lim => And (MeasureTheory.Memℒp f_lim p μ) (Filter.Tendsto (fun …
  -/
  let B1 n := ENNReal.ofReal (B n)
  have hB1_has : HasSum B1 (ENNReal.ofReal M) := by
    have h_tsum_B1 : ∑' i, B1 i = ENNReal.ofReal M := by
      change (∑' n : ℕ, ENNReal.ofReal (B n)) = ENNReal.ofReal M
      rw [← hB.tsum_eq]
      exact (ENNReal.ofReal_tsum_of_nonneg (fun n => le_of_lt (hB_pos n)) hB.summable).symm
    have h_sum := (@ENNReal.summable _ B1).hasSum
    rwa [h_tsum_B1] at h_sum
  have hB1 : ∑' i, B1 i < ∞ := by
    rw [hB1_has.tsum_eq]
    exact ENNReal.ofReal_lt_top
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hf : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (f n) (f m)) (B …
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    ⊢ Exists fun f_lim => And (MeasureTheory.Memℒp f_lim p μ) (Filter.Tendsto (fun …
  -/
  let f1 : ℕ → α → E := fun n => f n
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hf : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (f n) (f m)) (B …
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    ⊢ Exists fun f_lim => And (MeasureTheory.Memℒp f_lim p μ) (Filter.Tendsto (fun …
  -/
  refine H f1 (fun n => Lp.memℒp (f n)) B1 hB1 fun N n m hn hm => ?_
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hf : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (Dist.dist (f n) (f m)) (B …
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    N n m : Nat
    hn : LE.le N n
    hm : LE.le N m
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub (f1 n) (f1 m)) p μ) (B1 N)
  -/
  specialize hf N n m hn hm
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    N n m : Nat
    hn : LE.le N n
    hm : LE.le N m
    hf : LT.lt (Dist.dist (f n) (f m)) (B N)
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub (f1 n) (f1 m)) p μ) (B1 N)
  -/
  rw [dist_def] at hf
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    N n m : Nat
    hn : LE.le N n
    hm : LE.le N m
    hf : LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(f m)) p μ).toReal (B N)
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub (f1 n) (f1 m)) p μ) (B1 N)
  -/
  dsimp only [f1]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    N n m : Nat
    hn : LE.le N n
    hm : LE.le N m
    hf : LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(f m)) p μ).toReal (B N)
    ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(f m)) p μ) (B1 N)
  -/
  rwa [ENNReal.lt_ofReal_iff_toReal_lt]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    N n m : Nat
    hn : LE.le N n
    hm : LE.le N m
    hf : LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(f m)) p μ).toReal (B N)
    ⊢ Ne (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(f m)) p μ) Top.top
  -/
  rw [eLpNorm_congr_ae (Lp.coeFn_sub _ _).symm]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : Fact (LE.le 1 p)
    H : ∀ (f : Nat → α → E), (∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ) → ∀ (B : …
    B : Nat → Real := fun n => HPow.hPow (1 / 2) n
    hB_pos : ∀ (n : Nat), LT.lt 0 (B n)
    f : Nat → Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    M : Real
    hB : HasSum B M
    B1 : Nat → ENNReal := fun n => ENNReal.ofReal (B n)
    hB1_has : HasSum B1 (ENNReal.ofReal M)
    hB1 : LT.lt (tsum fun i => B1 i) Top.top
    f1 : Nat → α → E := fun n => ↑↑(f n)
    N n m : Nat
    hn : LE.le N n
    hm : LE.le N m
    hf : LT.lt (MeasureTheory.eLpNorm (HSub.hSub ↑↑(f n) ↑↑(f m)) p μ).toReal (B N)
    ⊢ Ne (MeasureTheory.eLpNorm (↑↑(HSub.hSub (f n) (f m))) p μ) Top.top
  -/
  exact Lp.eLpNorm_ne_top _
  /-
    🎉 no goals
  -/


private theorem eLpNorm'_sum_norm_sub_le_tsum_of_cauchy_eLpNorm' {f : ℕ → α → E}
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) {p : ℝ} (hp1 : 1 ≤ p) {B : ℕ → ℝ≥0∞}
    (h_cau : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm' (f n - f m) p μ < B N) (n : ℕ) :
    eLpNorm' (fun x => ∑ i ∈ Finset.range (n + 1), ‖f (i + 1) x - f i x‖) p μ ≤ ∑' i, B i := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    n : Nat
    ⊢ LE.le (MeasureTheory.eLpNorm' (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
  -/
  let f_norm_diff i x := ‖f (i + 1) x - f i x‖
  have hgf_norm_diff :
    ∀ n,
      (fun x => ∑ i ∈ Finset.range (n + 1), ‖f (i + 1) x - f i x‖) =
        ∑ i ∈ Finset.range (n + 1), f_norm_diff i :=
    fun n => funext fun x => by simp [f_norm_diff]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    n : Nat
    f_norm_diff : Nat → α → Real := fun i x => Norm.norm (HSub.hSub (f (HAdd.hAdd  …
    hgf_norm_diff : ∀ (n : Nat), Eq (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
    ⊢ LE.le (MeasureTheory.eLpNorm' (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
  -/
  rw [hgf_norm_diff]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    n : Nat
    f_norm_diff : Nat → α → Real := fun i x => Norm.norm (HSub.hSub (f (HAdd.hAdd  …
    hgf_norm_diff : ∀ (n : Nat), Eq (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
    ⊢ LE.le (MeasureTheory.eLpNorm' ((Finset.range (HAdd.hAdd n 1)).sum fun i => f …
  -/
  refine (eLpNorm'_sum_le (fun i _ => ((hf (i + 1)).sub (hf i)).norm) hp1).trans ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    n : Nat
    f_norm_diff : Nat → α → Real := fun i x => Norm.norm (HSub.hSub (f (HAdd.hAdd  …
    hgf_norm_diff : ∀ (n : Nat), Eq (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun i => MeasureTheory.eLpNorm' (f …
  -/
  simp_rw [eLpNorm'_norm]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    n : Nat
    f_norm_diff : Nat → α → Real := fun i x => Norm.norm (HSub.hSub (f (HAdd.hAdd  …
    hgf_norm_diff : ∀ (n : Nat), Eq (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
    ⊢ LE.le ((Finset.range (HAdd.hAdd n 1)).sum fun x => MeasureTheory.eLpNorm' (H …
  -/
  refine (Finset.sum_le_sum ?_).trans (sum_le_tsum _ (fun m _ => zero_le _) ENNReal.summable)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    n : Nat
    f_norm_diff : Nat → α → Real := fun i x => Norm.norm (HSub.hSub (f (HAdd.hAdd  …
    hgf_norm_diff : ∀ (n : Nat), Eq (fun x => (Finset.range (HAdd.hAdd n 1)).sum f …
    ⊢ ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → LE.le (Measur …
  -/
  exact fun m _ => (h_cau m (m + 1) m (Nat.le_succ m) (le_refl m)).le
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm'_sum_norm_sub_le_tsum_of_cauchy_snorm' :=
  eLpNorm'_sum_norm_sub_le_tsum_of_cauchy_eLpNorm'


private theorem lintegral_rpow_sum_coe_nnnorm_sub_le_rpow_tsum
    {f : ℕ → α → E} {p : ℝ} (hp1 : 1 ≤ p) {B : ℕ → ℝ≥0∞} (n : ℕ)
    (hn : eLpNorm' (fun x => ∑ i ∈ Finset.range (n + 1), ‖f (i + 1) x - f i x‖) p μ ≤ ∑' i, B i) :
    (∫⁻ a, (∑ i ∈ Finset.range (n + 1), ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ≤
      (∑' i, B i) ^ p := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    n : Nat
    hn : LE.le (MeasureTheory.eLpNorm' (fun x => (Finset.range (HAdd.hAdd n 1)).su …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset.range (HAdd.hAd …
  -/
  have hp_pos : 0 < p := zero_lt_one.trans_le hp1
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    n : Nat
    hn : LE.le (MeasureTheory.eLpNorm' (fun x => (Finset.range (HAdd.hAdd n 1)).su …
    hp_pos : LT.lt 0 p
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset.range (HAdd.hAd …
  -/
  rw [← inv_inv p, @ENNReal.le_rpow_inv_iff _ _ p⁻¹ (by simp [hp_pos]), inv_inv p]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    n : Nat
    hn : LE.le (MeasureTheory.eLpNorm' (fun x => (Finset.range (HAdd.hAdd n 1)).su …
    hp_pos : LT.lt 0 p
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset.rang …
  -/
  simp_rw [eLpNorm'_eq_lintegral_nnnorm, one_div] at hn
  have h_nnnorm_nonneg :
    (fun a => (‖∑ i ∈ Finset.range (n + 1), ‖f (i + 1) a - f i a‖‖₊ : ℝ≥0∞) ^ p) = fun a =>
      (∑ i ∈ Finset.range (n + 1), (‖f (i + 1) a - f i a‖₊ : ℝ≥0∞)) ^ p := by
    ext1 a
    congr
    simp_rw [← ofReal_norm_eq_coe_nnnorm]
    rw [← ENNReal.ofReal_sum_of_nonneg]
    · rw [Real.norm_of_nonneg _]
      exact Finset.sum_nonneg fun x _ => norm_nonneg _
    · exact fun x _ => norm_nonneg _
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    n : Nat
    hp_pos : LT.lt 0 p
    hn : LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm. …
    h_nnnorm_nonneg : Eq (fun a => HPow.hPow (↑(NNNorm.nnnorm ((Finset.range (HAdd …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset.rang …
  -/
  rwa [h_nnnorm_nonneg] at hn
  /-
    🎉 no goals
  -/


private theorem lintegral_rpow_tsum_coe_nnnorm_sub_le_tsum {f : ℕ → α → E}
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) {p : ℝ} (hp1 : 1 ≤ p) {B : ℕ → ℝ≥0∞}
    (h :
      ∀ n,
        (∫⁻ a, (∑ i ∈ Finset.range (n + 1), ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ≤
          (∑' i, B i) ^ p) :
    (∫⁻ a, (∑' i, ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ^ (1 / p) ≤ ∑' i, B i := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset. …
    ⊢ LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (tsum fun i = …
  -/
  have hp_pos : 0 < p := zero_lt_one.trans_le hp1
  suffices h_pow : (∫⁻ a, (∑' i, ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ≤ (∑' i, B i) ^ p by
      rwa [one_div, ← ENNReal.le_rpow_inv_iff (by simp [hp_pos] : 0 < p⁻¹), inv_inv]
  have h_tsum_1 :
    ∀ g : ℕ → ℝ≥0∞, ∑' i, g i = atTop.liminf fun n => ∑ i ∈ Finset.range (n + 1), g i := by
    intro g
    rw [ENNReal.tsum_eq_liminf_sum_nat, ← liminf_nat_add _ 1]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset. …
    hp_pos : LT.lt 0 p
    h_tsum_1 : ∀ (g : Nat → ENNReal), Eq (tsum fun i => g i) (Filter.liminf (fun n …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (tsum fun i => ↑(NNNorm. …
  -/
  simp_rw [h_tsum_1 _]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset. …
    hp_pos : LT.lt 0 p
    h_tsum_1 : ∀ (g : Nat → ENNReal), Eq (tsum fun i => g i) (Filter.liminf (fun n …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (Filter.liminf (fun n => …
  -/
  rw [← h_tsum_1]
  have h_liminf_pow :
    (∫⁻ a, (atTop.liminf
      fun n => ∑ i ∈ Finset.range (n + 1), (‖f (i + 1) a - f i a‖₊ : ℝ≥0∞)) ^ p ∂μ) =
      ∫⁻ a, atTop.liminf
        fun n => (∑ i ∈ Finset.range (n + 1), (‖f (i + 1) a - f i a‖₊ : ℝ≥0∞)) ^ p ∂μ := by
    refine lintegral_congr fun x => ?_
    have h_rpow_mono := ENNReal.strictMono_rpow_of_pos (zero_lt_one.trans_le hp1)
    have h_rpow_surj := (ENNReal.rpow_left_bijective hp_pos.ne.symm).2
    refine (h_rpow_mono.orderIsoOfSurjective _ h_rpow_surj).liminf_apply ?_ ?_ ?_ ?_
    all_goals isBoundedDefault
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset. …
    hp_pos : LT.lt 0 p
    h_tsum_1 : ∀ (g : Nat → ENNReal), Eq (tsum fun i => g i) (Filter.liminf (fun n …
    h_liminf_pow : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (Filter.liminf …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow (Filter.liminf (fun n => …
  -/
  rw [h_liminf_pow]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    h : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset. …
    hp_pos : LT.lt 0 p
    h_tsum_1 : ∀ (g : Nat → ENNReal), Eq (tsum fun i => g i) (Filter.liminf (fun n …
    h_liminf_pow : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (Filter.liminf …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => Filter.liminf (fun n => HPow.hPow  …
  -/
  refine (lintegral_liminf_le' ?_).trans ?_
  · exact fun n =>
      (Finset.aemeasurable_sum (Finset.range (n + 1)) fun i _ =>
            ((hf (i + 1)).sub (hf i)).ennnorm).pow_const
        _
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      p : Real
      hp1 : LE.le 1 p
      B : Nat → ENNReal
      h : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun a => HPow.hPow ((Finset. …
      hp_pos : LT.lt 0 p
      h_tsum_1 : ∀ (g : Nat → ENNReal), Eq (tsum fun i => g i) (Filter.liminf (fun n …
      h_liminf_pow : Eq (MeasureTheory.lintegral μ fun a => HPow.hPow (Filter.liminf …
      ⊢ LE.le (Filter.liminf (fun n => MeasureTheory.lintegral μ fun a => HPow.hPow  …
    -/
  · exact liminf_le_of_frequently_le' (Frequently.of_forall h)
    /-
      🎉 no goals
    -/


private theorem tsum_nnnorm_sub_ae_lt_top {f : ℕ → α → E} (hf : ∀ n, AEStronglyMeasurable (f n) μ)
    {p : ℝ} (hp1 : 1 ≤ p) {B : ℕ → ℝ≥0∞} (hB : ∑' i, B i ≠ ∞)
    (h : (∫⁻ a, (∑' i, ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ^ (1 / p) ≤ ∑' i, B i) :
    ∀ᵐ x ∂μ, (∑' i, ‖f (i + 1) x - f i x‖₊ : ℝ≥0∞) < ∞ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h : LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (tsum fun i …
    ⊢ Filter.Eventually (fun x => LT.lt (tsum fun i => ↑(NNNorm.nnnorm (HSub.hSub  …
  -/
  have hp_pos : 0 < p := zero_lt_one.trans_le hp1
  have h_integral : (∫⁻ a, (∑' i, ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) < ∞ := by
    have h_tsum_lt_top : (∑' i, B i) ^ p < ∞ := ENNReal.rpow_lt_top_of_nonneg hp_pos.le hB
    refine lt_of_le_of_lt ?_ h_tsum_lt_top
    rwa [one_div, ← ENNReal.le_rpow_inv_iff (by simp [hp_pos] : 0 < p⁻¹), inv_inv] at h
  have rpow_ae_lt_top : ∀ᵐ x ∂μ, (∑' i, ‖f (i + 1) x - f i x‖₊ : ℝ≥0∞) ^ p < ∞ := by
    refine ae_lt_top' (AEMeasurable.pow_const ?_ _) h_integral.ne
    exact AEMeasurable.ennreal_tsum fun n => ((hf (n + 1)).sub (hf n)).ennnorm
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    p : Real
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h : LE.le (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (tsum fun i …
    hp_pos : LT.lt 0 p
    h_integral : LT.lt (MeasureTheory.lintegral μ fun a => HPow.hPow (tsum fun i = …
    rpow_ae_lt_top : Filter.Eventually (fun x => LT.lt (HPow.hPow (tsum fun i => ↑ …
    ⊢ Filter.Eventually (fun x => LT.lt (tsum fun i => ↑(NNNorm.nnnorm (HSub.hSub  …
  -/
  refine rpow_ae_lt_top.mono fun x hx => ?_
  rwa [← ENNReal.lt_rpow_inv_iff hp_pos,
    ENNReal.top_rpow_of_pos (by simp [hp_pos] : 0 < p⁻¹)] at hx


theorem ae_tendsto_of_cauchy_eLpNorm' [CompleteSpace E] {f : ℕ → α → E} {p : ℝ}
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hp1 : 1 ≤ p) {B : ℕ → ℝ≥0∞} (hB : ∑' i, B i ≠ ∞)
    (h_cau : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm' (f n - f m) p μ < B N) :
    ∀ᵐ x ∂μ, ∃ l : E, atTop.Tendsto (fun n => f n x) (𝓝 l) := by
  have h_summable : ∀ᵐ x ∂μ, Summable fun i : ℕ => f (i + 1) x - f i x := by
    have h1 :
      ∀ n, eLpNorm' (fun x => ∑ i ∈ Finset.range (n + 1), ‖f (i + 1) x - f i x‖) p μ ≤ ∑' i, B i :=
      eLpNorm'_sum_norm_sub_le_tsum_of_cauchy_eLpNorm' hf hp1 h_cau
    have h2 :
      ∀ n,
        (∫⁻ a, (∑ i ∈ Finset.range (n + 1), ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ≤
          (∑' i, B i) ^ p :=
      fun n => lintegral_rpow_sum_coe_nnnorm_sub_le_rpow_tsum hp1 n (h1 n)
    have h3 : (∫⁻ a, (∑' i, ‖f (i + 1) a - f i a‖₊ : ℝ≥0∞) ^ p ∂μ) ^ (1 / p) ≤ ∑' i, B i :=
      lintegral_rpow_tsum_coe_nnnorm_sub_le_tsum hf hp1 h2
    have h4 : ∀ᵐ x ∂μ, (∑' i, ‖f (i + 1) x - f i x‖₊ : ℝ≥0∞) < ∞ :=
      tsum_nnnorm_sub_ae_lt_top hf hp1 hB h3
    exact h4.mono fun x hx => .of_nnnorm <| ENNReal.tsum_coe_ne_top_iff_summable.mp hx.ne
  have h :
    ∀ᵐ x ∂μ, ∃ l : E,
      atTop.Tendsto (fun n => ∑ i ∈ Finset.range n, (f (i + 1) x - f i x)) (𝓝 l) := by
    refine h_summable.mono fun x hx => ?_
    let hx_sum := hx.hasSum.tendsto_sum_nat
    exact ⟨∑' i, (f (i + 1) x - f i x), hx_sum⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    p : Real
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    h_summable : Filter.Eventually (fun x => Summable fun i => HSub.hSub (f (HAdd. …
    h : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => (Fins …
    ⊢ Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => f n x)  …
  -/
  refine h.mono fun x hx => ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    p : Real
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    h_summable : Filter.Eventually (fun x => Summable fun i => HSub.hSub (f (HAdd. …
    h : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => (Fins …
    x : α
    hx : Exists fun l => Filter.Tendsto (fun n => (Finset.range n).sum fun i => HS …
    ⊢ Exists fun l => Filter.Tendsto (fun n => f n x) Filter.atTop (nhds l)
  -/
  cases' hx with l hx
  have h_rw_sum :
      (fun n => ∑ i ∈ Finset.range n, (f (i + 1) x - f i x)) = fun n => f n x - f 0 x := by
    ext1 n
    change
      (∑ i ∈ Finset.range n, ((fun m => f m x) (i + 1) - (fun m => f m x) i)) = f n x - f 0 x
    rw [Finset.sum_range_sub (fun m => f m x)]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    p : Real
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    h_summable : Filter.Eventually (fun x => Summable fun i => HSub.hSub (f (HAdd. …
    h : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => (Fins …
    x : α
    l : E
    hx : Filter.Tendsto (fun n => (Finset.range n).sum fun i => HSub.hSub (f (HAdd …
    h_rw_sum : Eq (fun n => (Finset.range n).sum fun i => HSub.hSub (f (HAdd.hAdd  …
    ⊢ Exists fun l => Filter.Tendsto (fun n => f n x) Filter.atTop (nhds l)
  -/
  rw [h_rw_sum] at hx
  have hf_rw : (fun n => f n x) = fun n => f n x - f 0 x + f 0 x := by
    ext1 n
    abel
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    p : Real
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    h_summable : Filter.Eventually (fun x => Summable fun i => HSub.hSub (f (HAdd. …
    h : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => (Fins …
    x : α
    l : E
    hx : Filter.Tendsto (fun n => HSub.hSub (f n x) (f 0 x)) Filter.atTop (nhds l)
    h_rw_sum : Eq (fun n => (Finset.range n).sum fun i => HSub.hSub (f (HAdd.hAdd  …
    hf_rw : Eq (fun n => f n x) fun n => HAdd.hAdd (HSub.hSub (f n x) (f 0 x)) (f  …
    ⊢ Exists fun l => Filter.Tendsto (fun n => f n x) Filter.atTop (nhds l)
  -/
  rw [hf_rw]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    p : Real
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp1 : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm' …
    h_summable : Filter.Eventually (fun x => Summable fun i => HSub.hSub (f (HAdd. …
    h : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => (Fins …
    x : α
    l : E
    hx : Filter.Tendsto (fun n => HSub.hSub (f n x) (f 0 x)) Filter.atTop (nhds l)
    h_rw_sum : Eq (fun n => (Finset.range n).sum fun i => HSub.hSub (f (HAdd.hAdd  …
    hf_rw : Eq (fun n => f n x) fun n => HAdd.hAdd (HSub.hSub (f n x) (f 0 x)) (f  …
    ⊢ Exists fun l => Filter.Tendsto (fun n => HAdd.hAdd (HSub.hSub (f n x) (f 0 x …
  -/
  exact ⟨l + f 0 x, Tendsto.add_const _ hx⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias ae_tendsto_of_cauchy_snorm' := ae_tendsto_of_cauchy_eLpNorm'


theorem ae_tendsto_of_cauchy_eLpNorm [CompleteSpace E] {f : ℕ → α → E}
    (hf : ∀ n, AEStronglyMeasurable (f n) μ) (hp : 1 ≤ p) {B : ℕ → ℝ≥0∞} (hB : ∑' i, B i ≠ ∞)
    (h_cau : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm (f n - f m) p μ < B N) :
    ∀ᵐ x ∂μ, ∃ l : E, atTop.Tendsto (fun n => f n x) (𝓝 l) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    ⊢ Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => f n x)  …
  -/
  by_cases hp_top : p = ∞
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : CompleteSpace E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      hp : LE.le 1 p
      B : Nat → ENNReal
      hB : Ne (tsum fun i => B i) Top.top
      h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
      hp_top : Eq p Top.top
      ⊢ Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => f n x)  …
    -/
  · simp_rw [hp_top] at *
    have h_cau_ae : ∀ᵐ x ∂μ, ∀ N n m, N ≤ n → N ≤ m → (‖(f n - f m) x‖₊ : ℝ≥0∞) < B N := by
      simp_rw [ae_all_iff]
      exact fun N n m hnN hmN => ae_lt_of_essSup_lt (h_cau N n m hnN hmN)
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : CompleteSpace E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      B : Nat → ENNReal
      hB : Ne (tsum fun i => B i) Top.top
      hp : LE.le 1 Top.top
      h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
      hp_top : True
      h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
      ⊢ Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => f n x)  …
    -/
    simp_rw [eLpNorm_exponent_top, eLpNormEssSup] at h_cau
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : CompleteSpace E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      B : Nat → ENNReal
      hB : Ne (tsum fun i => B i) Top.top
      hp : LE.le 1 Top.top
      hp_top : True
      h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
      h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (essSup (fun x => ENorm …
      ⊢ Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => f n x)  …
    -/
    refine h_cau_ae.mono fun x hx => cauchySeq_tendsto_of_complete ?_
    /-
      case pos
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : CompleteSpace E
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      B : Nat → ENNReal
      hB : Ne (tsum fun i => B i) Top.top
      hp : LE.le 1 Top.top
      hp_top : True
      h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
      h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (essSup (fun x => ENorm …
      x : α
      hx : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (↑(NNNorm.nnnorm (HSub.hSu …
      ⊢ CauchySeq fun n => f n x
    -/
    refine cauchySeq_of_le_tendsto_0 (fun n => (B n).toReal) ?_ ?_
      /-
        case pos.refine_1
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup E
        inst✝ : CompleteSpace E
        f : Nat → α → E
        hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
        B : Nat → ENNReal
        hB : Ne (tsum fun i => B i) Top.top
        hp : LE.le 1 Top.top
        hp_top : True
        h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
        h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (essSup (fun x => ENorm …
        x : α
        hx : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (↑(NNNorm.nnnorm (HSub.hSu …
        ⊢ ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n x) (f m x)) ( …
      -/
    · intro n m N hnN hmN
      /-
        case pos.refine_1
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup E
        inst✝ : CompleteSpace E
        f : Nat → α → E
        hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
        B : Nat → ENNReal
        hB : Ne (tsum fun i => B i) Top.top
        hp : LE.le 1 Top.top
        hp_top : True
        h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
        h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (essSup (fun x => ENorm …
        x : α
        hx : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (↑(NNNorm.nnnorm (HSub.hSu …
        n m N : Nat
        hnN : LE.le N n
        hmN : LE.le N m
        ⊢ LE.le (Dist.dist (f n x) (f m x)) ((fun n => (B n).toReal) N)
      -/
      specialize hx N n m hnN hmN
      rw [_root_.dist_eq_norm,
        ← ENNReal.ofReal_le_iff_le_toReal (ENNReal.ne_top_of_tsum_ne_top hB N),
        ofReal_norm_eq_coe_nnnorm]
      /-
        case pos.refine_1
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup E
        inst✝ : CompleteSpace E
        f : Nat → α → E
        hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
        B : Nat → ENNReal
        hB : Ne (tsum fun i => B i) Top.top
        hp : LE.le 1 Top.top
        hp_top : True
        h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
        h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (essSup (fun x => ENorm …
        x : α
        n m N : Nat
        hnN : LE.le N n
        hmN : LE.le N m
        hx : LT.lt (↑(NNNorm.nnnorm (HSub.hSub (f n) (f m) x))) (B N)
        ⊢ LE.le (↑(NNNorm.nnnorm (HSub.hSub (f n x) (f m x)))) (B N)
      -/
      exact hx.le
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        α : Type u_1
        E : Type u_2
        m0 : MeasurableSpace α
        p : ENNReal
        μ : MeasureTheory.Measure α
        inst✝¹ : NormedAddCommGroup E
        inst✝ : CompleteSpace E
        f : Nat → α → E
        hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
        B : Nat → ENNReal
        hB : Ne (tsum fun i => B i) Top.top
        hp : LE.le 1 Top.top
        hp_top : True
        h_cau_ae : Filter.Eventually (fun x => ∀ (N n m : Nat), LE.le N n → LE.le N m  …
        h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (essSup (fun x => ENorm …
        x : α
        hx : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (↑(NNNorm.nnnorm (HSub.hSu …
        ⊢ Filter.Tendsto (fun n => (B n).toReal) Filter.atTop (nhds 0)
      -/
    · rw [← ENNReal.zero_toReal]
      exact
        Tendsto.comp (g := ENNReal.toReal) (ENNReal.tendsto_toReal ENNReal.zero_ne_top)
          (ENNReal.tendsto_atTop_zero_of_tsum_ne_top hB)
  have hp1 : 1 ≤ p.toReal := by
    rw [← ENNReal.ofReal_le_iff_le_toReal hp_top, ENNReal.ofReal_one]
    exact hp
  have h_cau' : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm' (f n - f m) p.toReal μ < B N := by
    intro N n m hn hm
    specialize h_cau N n m hn hm
    rwa [eLpNorm_eq_eLpNorm' (zero_lt_one.trans_le hp).ne.symm hp_top] at h_cau
  /-
    case neg
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hp : LE.le 1 p
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    hp_top : Not (Eq p Top.top)
    hp1 : LE.le 1 p.toReal
    h_cau' : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm …
    ⊢ Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun n => f n x)  …
  -/
  exact ae_tendsto_of_cauchy_eLpNorm' hf hp1 hB h_cau'
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias ae_tendsto_of_cauchy_snorm := ae_tendsto_of_cauchy_eLpNorm


theorem cauchy_tendsto_of_tendsto {f : ℕ → α → E} (hf : ∀ n, AEStronglyMeasurable (f n) μ)
    (f_lim : α → E) {B : ℕ → ℝ≥0∞} (hB : ∑' i, B i ≠ ∞)
    (h_cau : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm (f n - f m) p μ < B N)
    (h_lim : ∀ᵐ x : α ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x))) :
    atTop.Tendsto (fun n => eLpNorm (f n - f_lim) p μ) (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) f_lim) p μ)  …
  -/
  rw [ENNReal.tendsto_atTop_zero]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ⊢ ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le  …
  -/
  intro ε hε
  have h_B : ∃ N : ℕ, B N ≤ ε := by
    suffices h_tendsto_zero : ∃ N : ℕ, ∀ n : ℕ, N ≤ n → B n ≤ ε from
      ⟨h_tendsto_zero.choose, h_tendsto_zero.choose_spec _ le_rfl⟩
    exact (ENNReal.tendsto_atTop_zero.mp (ENNReal.tendsto_atTop_zero_of_tsum_ne_top hB)) ε hε
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ε : ENNReal
    hε : GT.gt ε 0
    h_B : Exists fun N => LE.le (B N) ε
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  cases' h_B with N h_B
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ε : ENNReal
    hε : GT.gt ε 0
    N : Nat
    h_B : LE.le (B N) ε
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub. …
  -/
  refine ⟨N, fun n hn => ?_⟩
  have h_sub : eLpNorm (f n - f_lim) p μ ≤ atTop.liminf fun m => eLpNorm (f n - f m) p μ := by
    refine eLpNorm_lim_le_liminf_eLpNorm (fun m => (hf n).sub (hf m)) (f n - f_lim) ?_
    refine h_lim.mono fun x hx => ?_
    simp_rw [sub_eq_add_neg]
    exact Tendsto.add tendsto_const_nhds (Tendsto.neg hx)
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ε : ENNReal
    hε : GT.gt ε 0
    N : Nat
    h_B : LE.le (B N) ε
    n : Nat
    hn : GE.ge n N
    h_sub : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) f_lim) p μ) (Filter.limi …
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) f_lim) p μ) ε
  -/
  refine h_sub.trans ?_
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ε : ENNReal
    hε : GT.gt ε 0
    N : Nat
    h_B : LE.le (B N) ε
    n : Nat
    hn : GE.ge n N
    h_sub : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) f_lim) p μ) (Filter.limi …
    ⊢ LE.le (Filter.liminf (fun m => MeasureTheory.eLpNorm (HSub.hSub (f n) (f m)) …
  -/
  refine liminf_le_of_frequently_le' (frequently_atTop.mpr ?_)
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ε : ENNReal
    hε : GT.gt ε 0
    N : Nat
    h_B : LE.le (B N) ε
    n : Nat
    hn : GE.ge n N
    h_sub : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) f_lim) p μ) (Filter.limi …
    ⊢ ∀ (a : Nat), Exists fun b => And (GE.ge b a) (LE.le (MeasureTheory.eLpNorm ( …
  -/
  refine fun N1 => ⟨max N N1, le_max_right _ _, ?_⟩
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    f_lim : α → E
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    ε : ENNReal
    hε : GT.gt ε 0
    N : Nat
    h_B : LE.le (B N) ε
    n : Nat
    hn : GE.ge n N
    h_sub : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) f_lim) p μ) (Filter.limi …
    N1 : Nat
    ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub (f n) (f (Max.max N N1))) p μ) ε
  -/
  exact (h_cau N n (max N N1) hn (le_max_left _ _)).le.trans h_B
  /-
    🎉 no goals
  -/


theorem memℒp_of_cauchy_tendsto (hp : 1 ≤ p) {f : ℕ → α → E} (hf : ∀ n, Memℒp (f n) p μ)
    (f_lim : α → E) (h_lim_meas : AEStronglyMeasurable f_lim μ)
    (h_tendsto : atTop.Tendsto (fun n => eLpNorm (f n - f_lim) p μ) (𝓝 0)) : Memℒp f_lim p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) f_ …
    ⊢ MeasureTheory.Memℒp f_lim p μ
  -/
  refine ⟨h_lim_meas, ?_⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) f_ …
    ⊢ LT.lt (MeasureTheory.eLpNorm f_lim p μ) Top.top
  -/
  rw [ENNReal.tendsto_atTop_zero] at h_tendsto
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    ⊢ LT.lt (MeasureTheory.eLpNorm f_lim p μ) Top.top
  -/
  cases' h_tendsto 1 zero_lt_one with N h_tendsto_1
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    N : Nat
    h_tendsto_1 : ∀ (n : Nat), GE.ge n N → LE.le (MeasureTheory.eLpNorm (HSub.hSub …
    ⊢ LT.lt (MeasureTheory.eLpNorm f_lim p μ) Top.top
  -/
  specialize h_tendsto_1 N (le_refl N)
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    N : Nat
    h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
    ⊢ LT.lt (MeasureTheory.eLpNorm f_lim p μ) Top.top
  -/
  have h_add : f_lim = f_lim - f N + f N := by abel
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    N : Nat
    h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
    h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
    ⊢ LT.lt (MeasureTheory.eLpNorm f_lim p μ) Top.top
  -/
  rw [h_add]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    N : Nat
    h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
    h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
    ⊢ LT.lt (MeasureTheory.eLpNorm (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N)) p μ)  …
  -/
  refine lt_of_le_of_lt (eLpNorm_add_le (h_lim_meas.sub (hf N).1) (hf N).1 hp) ?_
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    N : Nat
    h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
    h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
    ⊢ LT.lt (HAdd.hAdd (MeasureTheory.eLpNorm (HSub.hSub f_lim (f N)) p μ) (Measur …
  -/
  rw [ENNReal.add_lt_top]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    f_lim : α → E
    h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
    h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
    N : Nat
    h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
    h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
    ⊢ And (LT.lt (MeasureTheory.eLpNorm (HSub.hSub f_lim (f N)) p μ) Top.top) (LT. …
  -/
  constructor
    /-
      case intro.left
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hp : LE.le 1 p
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      f_lim : α → E
      h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
      h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
      N : Nat
      h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
      h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
      ⊢ LT.lt (MeasureTheory.eLpNorm (HSub.hSub f_lim (f N)) p μ) Top.top
    -/
  · refine lt_of_le_of_lt ?_ ENNReal.one_lt_top
    /-
      case intro.left
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hp : LE.le 1 p
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      f_lim : α → E
      h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
      h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
      N : Nat
      h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
      h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
      ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub f_lim (f N)) p μ) 1
    -/
    have h_neg : f_lim - f N = -(f N - f_lim) := by simp
    /-
      case intro.left
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hp : LE.le 1 p
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      f_lim : α → E
      h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
      h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
      N : Nat
      h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
      h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
      h_neg : Eq (HSub.hSub f_lim (f N)) (Neg.neg (HSub.hSub (f N) f_lim))
      ⊢ LE.le (MeasureTheory.eLpNorm (HSub.hSub f_lim (f N)) p μ) 1
    -/
    rwa [h_neg, eLpNorm_neg]
    /-
      🎉 no goals
    -/
    /-
      case intro.right
      α : Type u_1
      E : Type u_2
      m0 : MeasurableSpace α
      p : ENNReal
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup E
      hp : LE.le 1 p
      f : Nat → α → E
      hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
      f_lim : α → E
      h_lim_meas : MeasureTheory.AEStronglyMeasurable f_lim μ
      h_tendsto : ∀ (ε : ENNReal), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n  …
      N : Nat
      h_tendsto_1 : LE.le (MeasureTheory.eLpNorm (HSub.hSub (f N) f_lim) p μ) 1
      h_add : Eq f_lim (HAdd.hAdd (HSub.hSub f_lim (f N)) (f N))
      ⊢ LT.lt (MeasureTheory.eLpNorm (f N) p μ) Top.top
    -/
  · exact (hf N).2
    /-
      🎉 no goals
    -/


theorem cauchy_complete_ℒp [CompleteSpace E] (hp : 1 ≤ p) {f : ℕ → α → E}
    (hf : ∀ n, Memℒp (f n) p μ) {B : ℕ → ℝ≥0∞} (hB : ∑' i, B i ≠ ∞)
    (h_cau : ∀ N n m : ℕ, N ≤ n → N ≤ m → eLpNorm (f n - f m) p μ < B N) :
    ∃ (f_lim : α → E), Memℒp f_lim p μ ∧
      atTop.Tendsto (fun n => eLpNorm (f n - f_lim) p μ) (𝓝 0) := by
  obtain ⟨f_lim, h_f_lim_meas, h_lim⟩ :
      ∃ f_lim : α → E, StronglyMeasurable f_lim ∧
        ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x)) :=
    exists_stronglyMeasurable_limit_of_tendsto_ae (fun n => (hf n).1)
      (ae_tendsto_of_cauchy_eLpNorm (fun n => (hf n).1) hp hB h_cau)
  have h_tendsto' : atTop.Tendsto (fun n => eLpNorm (f n - f_lim) p μ) (𝓝 0) :=
    cauchy_tendsto_of_tendsto (fun m => (hf m).1) f_lim hB h_cau h_lim
  have h_ℒp_lim : Memℒp f_lim p μ :=
    memℒp_of_cauchy_tendsto hp hf f_lim h_f_lim_meas.aestronglyMeasurable h_tendsto'
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    hp : LE.le 1 p
    f : Nat → α → E
    hf : ∀ (n : Nat), MeasureTheory.Memℒp (f n) p μ
    B : Nat → ENNReal
    hB : Ne (tsum fun i => B i) Top.top
    h_cau : ∀ (N n m : Nat), LE.le N n → LE.le N m → LT.lt (MeasureTheory.eLpNorm  …
    f_lim : α → E
    h_f_lim_meas : MeasureTheory.StronglyMeasurable f_lim
    h_lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atT …
    h_tendsto' : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) f …
    h_ℒp_lim : MeasureTheory.Memℒp f_lim p μ
    ⊢ Exists fun f_lim => And (MeasureTheory.Memℒp f_lim p μ) (Filter.Tendsto (fun …
  -/
  exact ⟨f_lim, h_ℒp_lim, h_tendsto'⟩
  /-
    🎉 no goals
  -/


instance instCompleteSpace [CompleteSpace E] [hp : Fact (1 ≤ p)] : CompleteSpace (Lp E p μ) :=
  completeSpace_lp_of_cauchy_complete_ℒp fun _f hf _B hB h_cau =>
    cauchy_complete_ℒp hp.elim hf hB.ne h_cau


/-- An additive subgroup of `Lp E p μ`, consisting of the equivalence classes which contain a
bounded continuous representative. -/
def MeasureTheory.Lp.boundedContinuousFunction : AddSubgroup (Lp E p μ) :=
  AddSubgroup.addSubgroupOf
    ((ContinuousMap.toAEEqFunAddHom μ).comp (toContinuousMapAddHom α E)).range (Lp E p μ)


/-- By definition, the elements of `Lp.boundedContinuousFunction E p μ` are the elements of
`Lp E p μ` which contain a bounded continuous representative. -/
theorem MeasureTheory.Lp.mem_boundedContinuousFunction_iff {f : Lp E p μ} :
    f ∈ MeasureTheory.Lp.boundedContinuousFunction E p μ ↔
      ∃ f₀ : α →ᵇ E, f₀.toContinuousMap.toAEEqFun μ = (f : α →ₘ[μ] E) :=
  AddSubgroup.mem_addSubgroupOf


/-- A bounded continuous function on a finite-measure space is in `Lp`. -/
theorem mem_Lp (f : α →ᵇ E) : f.toContinuousMap.toAEEqFun μ ∈ Lp E p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    ⊢ Membership.mem (MeasureTheory.Lp E p μ) (ContinuousMap.toAEEqFun μ f.toConti …
  -/
  refine Lp.mem_Lp_of_ae_bound ‖f‖ ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (↑(ContinuousMap.toAEEqFun μ f. …
  -/
  filter_upwards [f.toContinuousMap.coeFn_toAEEqFun μ] with x _
  /-
    case h
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    x : α
    a✝ : Eq (↑(ContinuousMap.toAEEqFun μ f.toContinuousMap) x) (f.toContinuousMap x)
    ⊢ LE.le (Norm.norm (↑(ContinuousMap.toAEEqFun μ f.toContinuousMap) x)) (Norm.n …
  -/
  convert f.norm_coe_le_norm x using 2
  /-
    🎉 no goals
  -/


/-- The `Lp`-norm of a bounded continuous function is at most a constant (depending on the measure
of the whole space) times its sup-norm. -/
theorem Lp_nnnorm_le (f : α →ᵇ E) :
    ‖(⟨f.toContinuousMap.toAEEqFun μ, mem_Lp f⟩ : Lp E p μ)‖₊ ≤
      measureUnivNNReal μ ^ p.toReal⁻¹ * ‖f‖₊ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    ⊢ LE.le (NNNorm.nnnorm ⟨ContinuousMap.toAEEqFun μ f.toContinuousMap, ⋯⟩) (HMul …
  -/
  apply Lp.nnnorm_le_of_ae_bound
  /-
    case hfC
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    ⊢ Filter.Eventually (fun x => LE.le (NNNorm.nnnorm (↑↑⟨ContinuousMap.toAEEqFun …
  -/
  refine (f.toContinuousMap.coeFn_toAEEqFun μ).mono ?_
  /-
    case hfC
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    ⊢ ∀ (x : α), Eq (↑(ContinuousMap.toAEEqFun μ f.toContinuousMap) x) (f.toContin …
  -/
  intro x hx
  /-
    case hfC
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    x : α
    hx : Eq (↑(ContinuousMap.toAEEqFun μ f.toContinuousMap) x) (f.toContinuousMap x)
    ⊢ LE.le (NNNorm.nnnorm (↑↑⟨ContinuousMap.toAEEqFun μ f.toContinuousMap, ⋯⟩ x)) …
  -/
  rw [← NNReal.coe_le_coe, coe_nnnorm, coe_nnnorm]
  /-
    case hfC
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : SecondCountableTopologyEither α E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction α E
    x : α
    hx : Eq (↑(ContinuousMap.toAEEqFun μ f.toContinuousMap) x) (f.toContinuousMap x)
    ⊢ LE.le (Norm.norm (↑↑⟨ContinuousMap.toAEEqFun μ f.toContinuousMap, ⋯⟩ x)) (No …
  -/
  convert f.norm_coe_le_norm x using 2
  /-
    🎉 no goals
  -/


/-- The `Lp`-norm of a bounded continuous function is at most a constant (depending on the measure
of the whole space) times its sup-norm. -/
theorem Lp_norm_le (f : α →ᵇ E) :
    ‖(⟨f.toContinuousMap.toAEEqFun μ, mem_Lp f⟩ : Lp E p μ)‖ ≤
      measureUnivNNReal μ ^ p.toReal⁻¹ * ‖f‖ :=
  Lp_nnnorm_le f


/-- The normed group homomorphism of considering a bounded continuous function on a finite-measure
space as an element of `Lp`. -/
def toLpHom [Fact (1 ≤ p)] : NormedAddGroupHom (α →ᵇ E) (Lp E p μ) :=
  { AddMonoidHom.codRestrict ((ContinuousMap.toAEEqFunAddHom μ).comp (toContinuousMapAddHom α E))
      (Lp E p μ) mem_Lp with
    bound' := ⟨_, Lp_norm_le⟩ }


theorem range_toLpHom [Fact (1 ≤ p)] :
    ((toLpHom p μ).range : AddSubgroup (Lp E p μ)) =
      MeasureTheory.Lp.boundedContinuousFunction E p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : SecondCountableTopologyEither α E
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : Fact (LE.le 1 p)
    ⊢ Eq (BoundedContinuousFunction.toLpHom p μ).range (MeasureTheory.Lp.boundedCo …
  -/
  symm
  convert AddMonoidHom.addSubgroupOf_range_eq_of_le
      ((ContinuousMap.toAEEqFunAddHom μ).comp (toContinuousMapAddHom α E))
      (by rintro - ⟨f, rfl⟩; exact mem_Lp f : _ ≤ Lp E p μ)


/-- The bounded linear map of considering a bounded continuous function on a finite-measure space
as an element of `Lp`. -/
def toLp [NormedField 𝕜] [NormedSpace 𝕜 E] : (α →ᵇ E) →L[𝕜] Lp E p μ :=
  LinearMap.mkContinuous
    (LinearMap.codRestrict (Lp.LpSubmodule E p μ 𝕜)
      ((ContinuousMap.toAEEqFunLinearMap μ).comp (toContinuousMapLinearMap α E 𝕜)) mem_Lp)
    _ Lp_norm_le


theorem coeFn_toLp [NormedField 𝕜] [NormedSpace 𝕜 E] (f : α →ᵇ E) :
    toLp (E := E) p μ 𝕜 f =ᵐ[μ] f :=
  AEEqFun.coeFn_mk f _


theorem range_toLp [NormedField 𝕜] [NormedSpace 𝕜 E] :
    (LinearMap.range (toLp p μ 𝕜 : (α →ᵇ E) →L[𝕜] Lp E p μ)).toAddSubgroup =
      MeasureTheory.Lp.boundedContinuousFunction E p μ :=
  range_toLpHom p μ


theorem toLp_norm_le [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E] :
    ‖(toLp p μ 𝕜 : (α →ᵇ E) →L[𝕜] Lp E p μ)‖ ≤ measureUnivNNReal μ ^ p.toReal⁻¹ :=
  LinearMap.mkContinuous_norm_le _ (measureUnivNNReal μ ^ p.toReal⁻¹).coe_nonneg _


theorem toLp_inj {f g : α →ᵇ E} [μ.IsOpenPosMeasure] [NormedField 𝕜] [NormedSpace 𝕜 E] :
    toLp (E := E) p μ 𝕜 f = toLp (E := E) p μ 𝕜 g ↔ f = g := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝³ : Fact (LE.le 1 p)
    f g : BoundedContinuousFunction α E
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    ⊢ Iff (Eq ((BoundedContinuousFunction.toLp p μ 𝕜) f) ((BoundedContinuousFuncti …
  -/
  refine ⟨fun h => ?_, by tauto⟩
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝³ : Fact (LE.le 1 p)
    f g : BoundedContinuousFunction α E
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    h : Eq ((BoundedContinuousFunction.toLp p μ 𝕜) f) ((BoundedContinuousFunction. …
    ⊢ Eq f g
  -/
  rw [← DFunLike.coe_fn_eq, ← (map_continuous f).ae_eq_iff_eq μ (map_continuous g)]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝³ : Fact (LE.le 1 p)
    f g : BoundedContinuousFunction α E
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    h : Eq ((BoundedContinuousFunction.toLp p μ 𝕜) f) ((BoundedContinuousFunction. …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑g
  -/
  refine (coeFn_toLp p μ 𝕜 f).symm.trans (EventuallyEq.trans ?_ <| coeFn_toLp p μ 𝕜 g)
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝³ : Fact (LE.le 1 p)
    f g : BoundedContinuousFunction α E
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    h : Eq ((BoundedContinuousFunction.toLp p μ 𝕜) f) ((BoundedContinuousFunction. …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp p μ 𝕜)  …
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem toLp_injective [μ.IsOpenPosMeasure] [NormedField 𝕜] [NormedSpace 𝕜 E] :
    Function.Injective (⇑(toLp p μ 𝕜 : (α →ᵇ E) →L[𝕜] Lp E p μ)) :=
  fun _f _g hfg => (toLp_inj μ).mp hfg


/-- The bounded linear map of considering a continuous function on a compact finite-measure
space `α` as an element of `Lp`.  By definition, the norm on `C(α, E)` is the sup-norm, transferred
from the space `α →ᵇ E` of bounded continuous functions, so this construction is just a matter of
transferring the structure from `BoundedContinuousFunction.toLp` along the isometry. -/
def toLp [NormedField 𝕜] [NormedSpace 𝕜 E] : C(α, E) →L[𝕜] Lp E p μ :=
  (BoundedContinuousFunction.toLp p μ 𝕜).comp
    (linearIsometryBoundedOfCompact α E 𝕜).toLinearIsometry.toContinuousLinearMap


theorem range_toLp [NormedField 𝕜] [NormedSpace 𝕜 E] :
    (LinearMap.range (toLp p μ 𝕜 : C(α, E) →L[𝕜] Lp E p μ)).toAddSubgroup =
      MeasureTheory.Lp.boundedContinuousFunction E p μ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : CompactSpace α
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    ⊢ Eq (LinearMap.range (ContinuousMap.toLp p μ 𝕜)).toAddSubgroup (MeasureTheory …
  -/
  refine SetLike.ext' ?_
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : CompactSpace α
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    ⊢ Eq ↑(LinearMap.range (ContinuousMap.toLp p μ 𝕜)).toAddSubgroup ↑(MeasureTheo …
  -/
  have := (linearIsometryBoundedOfCompact α E 𝕜).surjective
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : CompactSpace α
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    this : Function.Surjective ⇑(ContinuousMap.linearIsometryBoundedOfCompact α E 𝕜)
    ⊢ Eq ↑(LinearMap.range (ContinuousMap.toLp p μ 𝕜)).toAddSubgroup ↑(MeasureTheo …
  -/
  convert Function.Surjective.range_comp this (BoundedContinuousFunction.toLp (E := E) p μ 𝕜)
  rw [← BoundedContinuousFunction.range_toLp p μ (𝕜 := 𝕜), Submodule.coe_toAddSubgroup,
    LinearMap.range_coe]


theorem coeFn_toLp [NormedField 𝕜] [NormedSpace 𝕜 E] (f : C(α, E)) :
    toLp (E := E) p μ 𝕜 f =ᵐ[μ] f :=
  AEEqFun.coeFn_mk f _


theorem toLp_def [NormedField 𝕜] [NormedSpace 𝕜 E] (f : C(α, E)) :
    toLp (E := E) p μ 𝕜 f =
      BoundedContinuousFunction.toLp (E := E) p μ 𝕜 (linearIsometryBoundedOfCompact α E 𝕜 f) :=
  rfl


@[simp]
theorem toLp_comp_toContinuousMap [NormedField 𝕜] [NormedSpace 𝕜 E] (f : α →ᵇ E) :
    toLp (E := E) p μ 𝕜 f.toContinuousMap = BoundedContinuousFunction.toLp (E := E) p μ 𝕜 f :=
  rfl


@[simp]
theorem coe_toLp [NormedField 𝕜] [NormedSpace 𝕜 E] (f : C(α, E)) :
    (toLp (E := E) p μ 𝕜 f : α →ₘ[μ] E) = f.toAEEqFun μ :=
  rfl


theorem toLp_injective [μ.IsOpenPosMeasure] [NormedField 𝕜] [NormedSpace 𝕜 E] :
    Function.Injective (⇑(toLp p μ 𝕜 : C(α, E) →L[𝕜] Lp E p μ)) :=
  (BoundedContinuousFunction.toLp_injective _).comp (linearIsometryBoundedOfCompact α E 𝕜).injective


theorem toLp_inj {f g : C(α, E)} [μ.IsOpenPosMeasure] [NormedField 𝕜] [NormedSpace 𝕜 E] :
    toLp (E := E) p μ 𝕜 f = toLp (E := E) p μ 𝕜 g ↔ f = g :=
  (toLp_injective μ).eq_iff


/-- If a sum of continuous functions `g n` is convergent, and the same sum converges in `Lᵖ` to `h`,
then in fact `g n` converges uniformly to `h`. -/
theorem hasSum_of_hasSum_Lp {β : Type*} [μ.IsOpenPosMeasure] [NormedField 𝕜] [NormedSpace 𝕜 E]
    {g : β → C(α, E)} {f : C(α, E)} (hg : Summable g)
    (hg2 : HasSum (toLp (E := E) p μ 𝕜 ∘ g) (toLp (E := E) p μ 𝕜 f)) : HasSum g f := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : BorelSpace α
    inst✝⁶ : SecondCountableTopologyEither α E
    inst✝⁵ : CompactSpace α
    inst✝⁴ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝³ : Fact (LE.le 1 p)
    β : Type u_6
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    g : β → ContinuousMap α E
    f : ContinuousMap α E
    hg : Summable g
    hg2 : HasSum (Function.comp (⇑(ContinuousMap.toLp p μ 𝕜)) g) ((ContinuousMap.t …
    ⊢ HasSum g f
  -/
  convert Summable.hasSum hg
  /-
    case h.e'_6
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : TopologicalSpace α
    inst✝⁷ : BorelSpace α
    inst✝⁶ : SecondCountableTopologyEither α E
    inst✝⁵ : CompactSpace α
    inst✝⁴ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝³ : Fact (LE.le 1 p)
    β : Type u_6
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    g : β → ContinuousMap α E
    f : ContinuousMap α E
    hg : Summable g
    hg2 : HasSum (Function.comp (⇑(ContinuousMap.toLp p μ 𝕜)) g) ((ContinuousMap.t …
    ⊢ Eq f (tsum fun b => g b)
  -/
  exact toLp_injective μ (hg2.unique ((toLp p μ 𝕜).hasSum <| Summable.hasSum hg))
  /-
    🎉 no goals
  -/


theorem toLp_norm_eq_toLp_norm_coe :
    ‖(toLp p μ 𝕜 : C(α, E) →L[𝕜] Lp E p μ)‖ =
      ‖(BoundedContinuousFunction.toLp p μ 𝕜 : (α →ᵇ E) →L[𝕜] Lp E p μ)‖ :=
  ContinuousLinearMap.opNorm_comp_linearIsometryEquiv _ _


/-- Bound for the operator norm of `ContinuousMap.toLp`. -/
theorem toLp_norm_le :
    ‖(toLp p μ 𝕜 : C(α, E) →L[𝕜] Lp E p μ)‖ ≤ measureUnivNNReal μ ^ p.toReal⁻¹ := by
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : CompactSpace α
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    ⊢ LE.le (Norm.norm (ContinuousMap.toLp p μ 𝕜)) (HPow.hPow (↑(MeasureTheory.mea …
  -/
  rw [toLp_norm_eq_toLp_norm_coe]
  /-
    α : Type u_1
    E : Type u_2
    m0 : MeasurableSpace α
    p : ENNReal
    μ : MeasureTheory.Measure α
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : SecondCountableTopologyEither α E
    inst✝⁴ : CompactSpace α
    inst✝³ : MeasureTheory.IsFiniteMeasure μ
    𝕜 : Type u_5
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : NormedSpace 𝕜 E
    ⊢ LE.le (Norm.norm (BoundedContinuousFunction.toLp p μ 𝕜)) (HPow.hPow (↑(Measu …
  -/
  exact BoundedContinuousFunction.toLp_norm_le μ
  /-
    🎉 no goals
  -/


theorem pow_mul_meas_ge_le_norm (f : Lp E p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) (ε : ℝ≥0∞) :
    (ε * μ { x | ε ≤ (‖f x‖₊ : ℝ≥0∞) ^ p.toReal }) ^ (1 / p.toReal) ≤ ENNReal.ofReal ‖f‖ :=
  (ENNReal.ofReal_toReal (eLpNorm_ne_top f)).symm ▸
    pow_mul_meas_ge_le_eLpNorm μ hp_ne_zero hp_ne_top (Lp.aestronglyMeasurable f) ε


theorem mul_meas_ge_le_pow_norm (f : Lp E p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) (ε : ℝ≥0∞) :
    ε * μ { x | ε ≤ (‖f x‖₊ : ℝ≥0∞) ^ p.toReal } ≤ ENNReal.ofReal ‖f‖ ^ p.toReal :=
  (ENNReal.ofReal_toReal (eLpNorm_ne_top f)).symm ▸
    mul_meas_ge_le_pow_eLpNorm μ hp_ne_zero hp_ne_top (Lp.aestronglyMeasurable f) ε


/-- A version of Markov's inequality with elements of Lp. -/
theorem mul_meas_ge_le_pow_norm' (f : Lp E p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞)
    (ε : ℝ≥0∞) : ε ^ p.toReal * μ { x | ε ≤ ‖f x‖₊ } ≤ ENNReal.ofReal ‖f‖ ^ p.toReal :=
  (ENNReal.ofReal_toReal (eLpNorm_ne_top f)).symm ▸
    mul_meas_ge_le_pow_eLpNorm' μ hp_ne_zero hp_ne_top (Lp.aestronglyMeasurable f) ε


theorem meas_ge_le_mul_pow_norm (f : Lp E p μ) (hp_ne_zero : p ≠ 0) (hp_ne_top : p ≠ ∞) {ε : ℝ≥0∞}
    (hε : ε ≠ 0) : μ { x | ε ≤ ‖f x‖₊ } ≤ ε⁻¹ ^ p.toReal * ENNReal.ofReal ‖f‖ ^ p.toReal :=
  (ENNReal.ofReal_toReal (eLpNorm_ne_top f)).symm ▸
    meas_ge_le_mul_pow_eLpNorm μ hp_ne_zero hp_ne_top (Lp.aestronglyMeasurable f) hε


