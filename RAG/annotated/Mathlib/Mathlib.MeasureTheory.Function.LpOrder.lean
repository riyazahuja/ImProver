theorem coeFn_le (f g : Lp E p μ) : f ≤ᵐ[μ] g ↔ f ≤ g := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE ↑↑f ↑↑g) (LE.le f g)
  -/
  rw [← Subtype.coe_le_coe, ← AEEqFun.coeFn_le]
  /-
    🎉 no goals
  -/


theorem coeFn_nonneg (f : Lp E p μ) : 0 ≤ᵐ[μ] f ↔ 0 ≤ f := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE 0 ↑↑f) (LE.le 0 f)
  -/
  rw [← coeFn_le]
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE 0 ↑↑f) ((MeasureTheory.ae μ).Eventual …
  -/
  have h0 := Lp.coeFn_zero E p μ
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
    ⊢ Iff ((MeasureTheory.ae μ).EventuallyLE 0 ↑↑f) ((MeasureTheory.ae μ).Eventual …
  -/
  constructor <;> intro h <;> filter_upwards [h, h0] with _ _ h2
    /-
      case h
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝ : NormedLatticeAddCommGroup E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
      h : (MeasureTheory.ae μ).EventuallyLE 0 ↑↑f
      a✝¹ : α
      a✝ : LE.le (0 a✝¹) (↑↑f a✝¹)
      h2 : Eq (↑↑0 a✝¹) (0 a✝¹)
      ⊢ LE.le (↑↑0 a✝¹) (↑↑f a✝¹)
    -/
  · rwa [h2]
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝ : NormedLatticeAddCommGroup E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
      h0 : (MeasureTheory.ae μ).EventuallyEq (↑↑0) 0
      h : (MeasureTheory.ae μ).EventuallyLE ↑↑0 ↑↑f
      a✝¹ : α
      a✝ : LE.le (↑↑0 a✝¹) (↑↑f a✝¹)
      h2 : Eq (↑↑0 a✝¹) (0 a✝¹)
      ⊢ LE.le (0 a✝¹) (↑↑f a✝¹)
    -/
  · rwa [← h2]
    /-
      🎉 no goals
    -/


instance instAddLeftMono : AddLeftMono (Lp E p μ) := by
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    ⊢ AddLeftMono (Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x)
  -/
  refine ⟨fun f g₁ g₂ hg₁₂ => ?_⟩
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hg₁₂ : LE.le g₁ g₂
    ⊢ LE.le (HAdd.hAdd f g₁) (HAdd.hAdd f g₂)
  -/
  rw [← coeFn_le] at hg₁₂ ⊢
  /-
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑g₁ ↑↑g₂
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑(HAdd.hAdd f g₁) ↑↑(HAdd.hAdd f g₂)
  -/
  filter_upwards [coeFn_add f g₁, coeFn_add f g₂, hg₁₂] with _ h1 h2 h3
  /-
    case h
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑g₁ ↑↑g₂
    a✝ : α
    h1 : Eq (↑↑(HAdd.hAdd f g₁) a✝) (HAdd.hAdd (↑↑f) (↑↑g₁) a✝)
    h2 : Eq (↑↑(HAdd.hAdd f g₂) a✝) (HAdd.hAdd (↑↑f) (↑↑g₂) a✝)
    h3 : LE.le (↑↑g₁ a✝) (↑↑g₂ a✝)
    ⊢ LE.le (↑↑(HAdd.hAdd f g₁) a✝) (↑↑(HAdd.hAdd f g₂) a✝)
  -/
  rw [h1, h2, Pi.add_apply, Pi.add_apply]
  /-
    case h
    α : Type u_1
    E : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    p : ENNReal
    inst✝ : NormedLatticeAddCommGroup E
    f g₁ g₂ : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    hg₁₂ : (MeasureTheory.ae μ).EventuallyLE ↑↑g₁ ↑↑g₂
    a✝ : α
    h1 : Eq (↑↑(HAdd.hAdd f g₁) a✝) (HAdd.hAdd (↑↑f) (↑↑g₁) a✝)
    h2 : Eq (↑↑(HAdd.hAdd f g₂) a✝) (HAdd.hAdd (↑↑f) (↑↑g₂) a✝)
    h3 : LE.le (↑↑g₁ a✝) (↑↑g₂ a✝)
    ⊢ LE.le (HAdd.hAdd (↑↑f a✝) (↑↑g₁ a✝)) (HAdd.hAdd (↑↑f a✝) (↑↑g₂ a✝))
  -/
  exact add_le_add le_rfl h3
  /-
    🎉 no goals
  -/


instance instOrderedAddCommGroup : OrderedAddCommGroup (Lp E p μ) :=
  { Subtype.partialOrder _, AddSubgroup.toAddCommGroup _ with
    add_le_add_left := fun _ _ => add_le_add_left }


theorem _root_.MeasureTheory.Memℒp.sup {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    Memℒp (f ⊔ g) p μ :=
  Memℒp.mono' (hf.norm.add hg.norm) (hf.1.sup hg.1)
    (Filter.Eventually.of_forall fun x => norm_sup_le_add (f x) (g x))


theorem _root_.MeasureTheory.Memℒp.inf {f g : α → E} (hf : Memℒp f p μ) (hg : Memℒp g p μ) :
    Memℒp (f ⊓ g) p μ :=
  Memℒp.mono' (hf.norm.add hg.norm) (hf.1.inf hg.1)
    (Filter.Eventually.of_forall fun x => norm_inf_le_add (f x) (g x))


theorem _root_.MeasureTheory.Memℒp.abs {f : α → E} (hf : Memℒp f p μ) : Memℒp |f| p μ :=
  hf.sup hf.neg


instance instLattice : Lattice (Lp E p μ) :=
  Subtype.lattice
    (fun f g hf hg => by
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝ : NormedLatticeAddCommGroup E
        f g : MeasureTheory.AEEqFun α E μ
        hf : Membership.mem (MeasureTheory.Lp E p μ) f
        hg : Membership.mem (MeasureTheory.Lp E p μ) g
        ⊢ Membership.mem (MeasureTheory.Lp E p μ) (Max.max f g)
      -/
      rw [mem_Lp_iff_memℒp] at *
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝ : NormedLatticeAddCommGroup E
        f g : MeasureTheory.AEEqFun α E μ
        hf : MeasureTheory.Memℒp (↑f) p μ
        hg : MeasureTheory.Memℒp (↑g) p μ
        ⊢ MeasureTheory.Memℒp (↑(Max.max f g)) p μ
      -/
      exact (memℒp_congr_ae (AEEqFun.coeFn_sup _ _)).mpr (hf.sup hg))
      /-
        🎉 no goals
      -/
    fun f g hf hg => by
    /-
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝ : NormedLatticeAddCommGroup E
      f g : MeasureTheory.AEEqFun α E μ
      hf : Membership.mem (MeasureTheory.Lp E p μ) f
      hg : Membership.mem (MeasureTheory.Lp E p μ) g
      ⊢ Membership.mem (MeasureTheory.Lp E p μ) (Min.min f g)
    -/
    rw [mem_Lp_iff_memℒp] at *
    /-
      α : Type u_1
      E : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      p : ENNReal
      inst✝ : NormedLatticeAddCommGroup E
      f g : MeasureTheory.AEEqFun α E μ
      hf : MeasureTheory.Memℒp (↑f) p μ
      hg : MeasureTheory.Memℒp (↑g) p μ
      ⊢ MeasureTheory.Memℒp (↑(Min.min f g)) p μ
    -/
    exact (memℒp_congr_ae (AEEqFun.coeFn_inf _ _)).mpr (hf.inf hg)
    /-
      🎉 no goals
    -/


theorem coeFn_sup (f g : Lp E p μ) : ⇑(f ⊔ g) =ᵐ[μ] ⇑f ⊔ ⇑g :=
  AEEqFun.coeFn_sup _ _


theorem coeFn_inf (f g : Lp E p μ) : ⇑(f ⊓ g) =ᵐ[μ] ⇑f ⊓ ⇑g :=
  AEEqFun.coeFn_inf _ _


theorem coeFn_abs (f : Lp E p μ) : ⇑|f| =ᵐ[μ] fun x => |f x| :=
  AEEqFun.coeFn_abs _


noncomputable instance instNormedLatticeAddCommGroup [Fact (1 ≤ p)] :
    NormedLatticeAddCommGroup (Lp E p μ) :=
  { Lp.instLattice, Lp.instNormedAddCommGroup with
    add_le_add_left := fun _ _ => add_le_add_left
    solid := fun f g hfg => by
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝¹ : NormedLatticeAddCommGroup E
        inst✝ : Fact (LE.le 1 p)
        f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hfg : LE.le (abs f) (abs g)
        ⊢ LE.le (Norm.norm f) (Norm.norm g)
      -/
      rw [← coeFn_le] at hfg
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝¹ : NormedLatticeAddCommGroup E
        inst✝ : Fact (LE.le 1 p)
        f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hfg : (MeasureTheory.ae μ).EventuallyLE ↑↑(abs f) ↑↑(abs g)
        ⊢ LE.le (Norm.norm f) (Norm.norm g)
      -/
      simp_rw [Lp.norm_def, ENNReal.toReal_le_toReal (Lp.eLpNorm_ne_top f) (Lp.eLpNorm_ne_top g)]
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝¹ : NormedLatticeAddCommGroup E
        inst✝ : Fact (LE.le 1 p)
        f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hfg : (MeasureTheory.ae μ).EventuallyLE ↑↑(abs f) ↑↑(abs g)
        ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p μ) (MeasureTheory.eLpNorm (↑↑g) p μ)
      -/
      refine eLpNorm_mono_ae ?_
      /-
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝¹ : NormedLatticeAddCommGroup E
        inst✝ : Fact (LE.le 1 p)
        f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hfg : (MeasureTheory.ae μ).EventuallyLE ↑↑(abs f) ↑↑(abs g)
        ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (↑↑f x)) (Norm.norm (↑↑g x))) ( …
      -/
      filter_upwards [hfg, Lp.coeFn_abs f, Lp.coeFn_abs g] with x hx hxf hxg
      /-
        case h
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝¹ : NormedLatticeAddCommGroup E
        inst✝ : Fact (LE.le 1 p)
        f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hfg : (MeasureTheory.ae μ).EventuallyLE ↑↑(abs f) ↑↑(abs g)
        x : α
        hx : LE.le (↑↑(abs f) x) (↑↑(abs g) x)
        hxf : Eq (↑↑(abs f) x) (abs (↑↑f x))
        hxg : Eq (↑↑(abs g) x) (abs (↑↑g x))
        ⊢ LE.le (Norm.norm (↑↑f x)) (Norm.norm (↑↑g x))
      -/
      rw [hxf, hxg] at hx
      /-
        case h
        α : Type u_1
        E : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        p : ENNReal
        inst✝¹ : NormedLatticeAddCommGroup E
        inst✝ : Fact (LE.le 1 p)
        f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
        hfg : (MeasureTheory.ae μ).EventuallyLE ↑↑(abs f) ↑↑(abs g)
        x : α
        hx : LE.le (abs (↑↑f x)) (abs (↑↑g x))
        hxf : Eq (↑↑(abs f) x) (abs (↑↑f x))
        hxg : Eq (↑↑(abs g) x) (abs (↑↑g x))
        ⊢ LE.le (Norm.norm (↑↑f x)) (Norm.norm (↑↑g x))
      -/
      exact HasSolidNorm.solid hx }
      /-
        🎉 no goals
      -/


