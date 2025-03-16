/-- If `Γ` is linearly ordered and `R` has zero, then `HahnSeries Γ R` consists of
  formal series over `Γ` with coefficients in `R`, whose supports are well-founded. -/
@[ext]
structure HahnSeries (Γ : Type*) (R : Type*) [PartialOrder Γ] [Zero R] where
  /-- The coefficient function of a Hahn Series. -/
  coeff : Γ → R
  isPWO_support' : (Function.support coeff).IsPWO


theorem coeff_injective : Injective (coeff : HahnSeries Γ R → Γ → R) :=
  fun _ _ => HahnSeries.ext


@[simp]
theorem coeff_inj {x y : HahnSeries Γ R} : x.coeff = y.coeff ↔ x = y :=
  coeff_injective.eq_iff


/-- The support of a Hahn series is just the set of indices whose coefficients are nonzero.
  Notably, it is well-founded. -/
nonrec def support (x : HahnSeries Γ R) : Set Γ :=
  support x.coeff


@[simp]
theorem isPWO_support (x : HahnSeries Γ R) : x.support.IsPWO :=
  x.isPWO_support'


@[simp]
theorem isWF_support (x : HahnSeries Γ R) : x.support.IsWF :=
  x.isPWO_support.isWF


@[simp]
theorem mem_support (x : HahnSeries Γ R) (a : Γ) : a ∈ x.support ↔ x.coeff a ≠ 0 :=
  Iff.refl _


instance : Zero (HahnSeries Γ R) :=
  ⟨{  coeff := 0
                           /-
                             Γ : Type u_1
                             Γ' : Type u_2
                             R : Type u_3
                             S : Type u_4
                             inst✝¹ : PartialOrder Γ
                             inst✝ : Zero R
                             ⊢ (Function.support 0).IsPWO
                           -/
      isPWO_support' := by simp }⟩
                           /-
                             🎉 no goals
                           -/


instance : Inhabited (HahnSeries Γ R) :=
  ⟨0⟩


instance [Subsingleton R] : Subsingleton (HahnSeries Γ R) :=
                                 /-
                                   Γ : Type u_1
                                   Γ' : Type u_2
                                   R : Type u_3
                                   S : Type u_4
                                   inst✝² : PartialOrder Γ
                                   inst✝¹ : Zero R
                                   inst✝ : Subsingleton R
                                   x✝¹ x✝ : HahnSeries Γ R
                                   ⊢ Eq x✝¹.coeff x✝.coeff
                                 -/
  ⟨fun _ _ => HahnSeries.ext (by subsingleton)⟩
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem zero_coeff {a : Γ} : (0 : HahnSeries Γ R).coeff a = 0 :=
  rfl


@[simp]
theorem coeff_fun_eq_zero_iff {x : HahnSeries Γ R} : x.coeff = 0 ↔ x = 0 :=
  coeff_injective.eq_iff' rfl


theorem ne_zero_of_coeff_ne_zero {x : HahnSeries Γ R} {g : Γ} (h : x.coeff g ≠ 0) : x ≠ 0 :=
  mt (fun x0 => (x0.symm ▸ zero_coeff : x.coeff g = 0)) h


@[simp]
theorem support_zero : support (0 : HahnSeries Γ R) = ∅ :=
  Function.support_zero


@[simp]
nonrec theorem support_nonempty_iff {x : HahnSeries Γ R} : x.support.Nonempty ↔ x ≠ 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    ⊢ Iff x.support.Nonempty (Ne x 0)
  -/
  rw [support, support_nonempty_iff, Ne, coeff_fun_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_eq_empty_iff {x : HahnSeries Γ R} : x.support = ∅ ↔ x = 0 :=
  Function.support_eq_empty_iff.trans coeff_fun_eq_zero_iff


/-- The map of Hahn series induced by applying a zero-preserving map to each coefficient. -/
@[simps]
def map [Zero S] (x : HahnSeries Γ R) {F : Type*} [FunLike F R S] [ZeroHomClass F R S] (f : F) :
    HahnSeries Γ S where
  coeff g := f (x.coeff g)
  isPWO_support' := x.isPWO_support.mono <| Function.support_comp_subset (ZeroHomClass.map_zero f) _


@[simp]
protected lemma map_zero [Zero S] (f : ZeroHom R S) :
    (0 : HahnSeries Γ R).map f = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero S
    f : ZeroHom R S
    ⊢ Eq (HahnSeries.map 0 f) 0
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- Change a HahnSeries with coefficients in HahnSeries to a HahnSeries on the Lex product. -/
def ofIterate [PartialOrder Γ'] (x : HahnSeries Γ (HahnSeries Γ' R)) :
    HahnSeries (Γ ×ₗ Γ') R where
  coeff := fun g => coeff (coeff x g.1) g.2
  isPWO_support' := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      x : HahnSeries Γ (HahnSeries Γ' R)
      ⊢ (Function.support fun g => (x.coeff g.1).coeff g.2).IsPWO
    -/
    refine Set.PartiallyWellOrderedOn.subsetProdLex ?_ ?_
      /-
        case refine_1
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        inst✝² : PartialOrder Γ
        inst✝¹ : Zero R
        inst✝ : PartialOrder Γ'
        x : HahnSeries Γ (HahnSeries Γ' R)
        ⊢ (Set.image (fun x => (ofLex x).1) (Function.support fun g => (x.coeff g.1).c …
      -/
    · refine Set.IsPWO.mono x.isPWO_support' ?_
      /-
        case refine_1
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        inst✝² : PartialOrder Γ
        inst✝¹ : Zero R
        inst✝ : PartialOrder Γ'
        x : HahnSeries Γ (HahnSeries Γ' R)
        ⊢ HasSubset.Subset (Set.image (fun x => (ofLex x).1) (Function.support fun g = …
      -/
      simp_rw [Set.image_subset_iff, support_subset_iff, Set.mem_preimage, Function.mem_support]
      /-
        case refine_1
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        inst✝² : PartialOrder Γ
        inst✝¹ : Zero R
        inst✝ : PartialOrder Γ'
        x : HahnSeries Γ (HahnSeries Γ' R)
        ⊢ ∀ (x_1 : Lex (Prod Γ Γ')), Ne ((x.coeff x_1.1).coeff x_1.2) 0 → Ne (x.coeff  …
      -/
      exact fun _ ↦ ne_zero_of_coeff_ne_zero
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        inst✝² : PartialOrder Γ
        inst✝¹ : Zero R
        inst✝ : PartialOrder Γ'
        x : HahnSeries Γ (HahnSeries Γ' R)
        ⊢ ∀ (a : Γ), (setOf fun y => Membership.mem (Function.support fun g => (x.coef …
      -/
    · exact fun a => by simpa [Function.mem_support, ne_eq] using (x.coeff a).isPWO_support'
      /-
        🎉 no goals
      -/


@[simp]
lemma mk_eq_zero (f : Γ → R) (h) : HahnSeries.mk f h = 0 ↔ f = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    f : Γ → R
    h : (Function.support f).IsPWO
    ⊢ Iff (Eq { coeff := f, isPWO_support' := h } 0) (Eq f 0)
  -/
  rw [HahnSeries.ext_iff]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    f : Γ → R
    h : (Function.support f).IsPWO
    ⊢ Iff (Eq { coeff := f, isPWO_support' := h }.coeff (HahnSeries.coeff 0)) (Eq  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Change a Hahn series on a lex product to a Hahn series with coefficients in a Hahn series. -/
def toIterate [PartialOrder Γ'] (x : HahnSeries (Γ ×ₗ Γ') R) :
    HahnSeries Γ (HahnSeries Γ' R) where
  coeff := fun g => {
    coeff := fun g' => coeff x (g, g')
    isPWO_support' := Set.PartiallyWellOrderedOn.fiberProdLex x.isPWO_support' g
  }
  isPWO_support' := by
    have h₁ : (Function.support fun g => HahnSeries.mk (fun g' => x.coeff (g, g'))
        (Set.PartiallyWellOrderedOn.fiberProdLex x.isPWO_support' g)) = Function.support
        fun g => fun g' => x.coeff (g, g') := by
      simp only [Function.support, ne_eq, mk_eq_zero]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      x : HahnSeries (Lex (Prod Γ Γ')) R
      h₁ : Eq (Function.support fun g => { coeff := fun g' => x.coeff { fst := g, sn …
      ⊢ (Function.support fun g => { coeff := fun g' => x.coeff { fst := g, snd := g …
    -/
    rw [h₁, Function.support_curry' x.coeff]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      x : HahnSeries (Lex (Prod Γ Γ')) R
      h₁ : Eq (Function.support fun g => { coeff := fun g' => x.coeff { fst := g, sn …
      ⊢ (Set.image Prod.fst (Function.support x.coeff)).IsPWO
    -/
    exact Set.PartiallyWellOrderedOn.imageProdLex x.isPWO_support'
    /-
      🎉 no goals
    -/


/-- The equivalence between iterated Hahn series and Hahn series on the lex product. -/
@[simps]
def iterateEquiv [PartialOrder Γ'] :
    HahnSeries Γ (HahnSeries Γ' R) ≃ HahnSeries (Γ ×ₗ Γ') R where
  toFun := ofIterate
  invFun := toIterate
  left_inv := congrFun rfl
  right_inv := congrFun rfl


open Classical in
/-- `single a r` is the Hahn series which has coefficient `r` at `a` and zero otherwise. -/
def single (a : Γ) : ZeroHom R (HahnSeries Γ R) where
  toFun r :=
    { coeff := Pi.single a r
      isPWO_support' := (Set.isPWO_singleton a).mono Pi.support_single_subset }
  map_zero' := HahnSeries.ext (Pi.single_zero _)


@[simp]
theorem single_coeff_same (a : Γ) (r : R) : (single a r).coeff a = r := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a : Γ
    r : R
    ⊢ Eq (((HahnSeries.single a) r).coeff a) r
  -/
  classical exact Pi.single_eq_same (f := fun _ => R) a r
  /-
    🎉 no goals
  -/


@[simp]
theorem single_coeff_of_ne (h : b ≠ a) : (single a r).coeff b = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a b : Γ
    r : R
    h : Ne b a
    ⊢ Eq (((HahnSeries.single a) r).coeff b) 0
  -/
  classical exact Pi.single_eq_of_ne (f := fun _ => R) h r
  /-
    🎉 no goals
  -/


open Classical in
theorem single_coeff : (single a r).coeff b = if b = a then r else 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a b : Γ
    r : R
    ⊢ Eq (((HahnSeries.single a) r).coeff b) (ite (Eq b a) r 0)
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem support_single_of_ne (h : r ≠ 0) : support (single a r) = {a} := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a : Γ
    r : R
    h : Ne r 0
    ⊢ Eq ((HahnSeries.single a) r).support (Singleton.singleton a)
  -/
  classical exact Pi.support_single_of_ne h
  /-
    🎉 no goals
  -/


theorem support_single_subset : support (single a r) ⊆ {a} := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a : Γ
    r : R
    ⊢ HasSubset.Subset ((HahnSeries.single a) r).support (Singleton.singleton a)
  -/
  classical exact Pi.support_single_subset
  /-
    🎉 no goals
  -/


theorem eq_of_mem_support_single {b : Γ} (h : b ∈ support (single a r)) : b = a :=
  support_single_subset h


theorem single_eq_zero : single a (0 : R) = 0 :=
  (single a).map_zero


theorem single_injective (a : Γ) : Function.Injective (single a : R → HahnSeries Γ R) :=
                   /-
                     Γ : Type u_1
                     R : Type u_3
                     inst✝¹ : PartialOrder Γ
                     inst✝ : Zero R
                     a : Γ
                     r s : R
                     rs : Eq ((HahnSeries.single a) r) ((HahnSeries.single a) s)
                     ⊢ Eq r s
                   -/
  fun r s rs => by rw [← single_coeff_same a r, ← single_coeff_same a s, rs]
                   /-
                     🎉 no goals
                   -/


theorem single_ne_zero (h : r ≠ 0) : single a r ≠ 0 := fun con =>
  h (single_injective a (con.trans single_eq_zero.symm))


@[simp]
theorem single_eq_zero_iff {a : Γ} {r : R} : single a r = 0 ↔ r = 0 :=
  map_eq_zero_iff _ <| single_injective a


@[simp]
protected lemma map_single [Zero S] (f : ZeroHom R S) : (single a r).map f = single a (f r) := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    a : Γ
    r : R
    inst✝ : Zero S
    f : ZeroHom R S
    ⊢ Eq (((HahnSeries.single a) r).map f) ((HahnSeries.single a) (f r))
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    a : Γ
    r : R
    inst✝ : Zero S
    f : ZeroHom R S
    g : Γ
    ⊢ Eq ((((HahnSeries.single a) r).map f).coeff g) (((HahnSeries.single a) (f r) …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : g = a <;> simp [h]
                         /-
                           🎉 no goals
                         -/


instance [Nonempty Γ] [Nontrivial R] : Nontrivial (HahnSeries Γ R) :=
  ⟨by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝³ : PartialOrder Γ
      inst✝² : Zero R
      a b : Γ
      r : R
      inst✝¹ : Nonempty Γ
      inst✝ : Nontrivial R
      ⊢ Exists fun x => Exists fun y => Ne x y
    -/
    obtain ⟨r, s, rs⟩ := exists_pair_ne R
    /-
      case intro.intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝³ : PartialOrder Γ
      inst✝² : Zero R
      a b : Γ
      r✝ : R
      inst✝¹ : Nonempty Γ
      inst✝ : Nontrivial R
      r s : R
      rs : Ne r s
      ⊢ Exists fun x => Exists fun y => Ne x y
    -/
    inhabit Γ
    /-
      case intro.intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝³ : PartialOrder Γ
      inst✝² : Zero R
      a b : Γ
      r✝ : R
      inst✝¹ : Nonempty Γ
      inst✝ : Nontrivial R
      r s : R
      rs : Ne r s
      inhabited_h : Inhabited Γ
      ⊢ Exists fun x => Exists fun y => Ne x y
    -/
    refine ⟨single default r, single default s, fun con => rs ?_⟩
    /-
      case intro.intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      inst✝³ : PartialOrder Γ
      inst✝² : Zero R
      a b : Γ
      r✝ : R
      inst✝¹ : Nonempty Γ
      inst✝ : Nontrivial R
      r s : R
      rs : Ne r s
      inhabited_h : Inhabited Γ
      con : Eq ((HahnSeries.single Inhabited.default) r) ((HahnSeries.single Inhabit …
      ⊢ Eq r s
    -/
    rw [← single_coeff_same (default : Γ) r, con, single_coeff_same]⟩
    /-
      🎉 no goals
    -/


open Classical in
/-- The orderTop of a Hahn series `x` is a minimal element of `WithTop Γ` where `x` has a nonzero
coefficient if `x ≠ 0`, and is `⊤` when `x = 0`. -/
def orderTop (x : HahnSeries Γ R) : WithTop Γ :=
  if h : x = 0 then ⊤ else x.isWF_support.min (support_nonempty_iff.2 h)


@[simp]
theorem orderTop_zero : orderTop (0 : HahnSeries Γ R) = ⊤ :=
  dif_pos rfl


theorem orderTop_of_ne {x : HahnSeries Γ R} (hx : x ≠ 0) :
    orderTop x = x.isWF_support.min (support_nonempty_iff.2 hx) :=
  dif_neg hx


@[simp]
theorem ne_zero_iff_orderTop {x : HahnSeries Γ R} : x ≠ 0 ↔ orderTop x ≠ ⊤ := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    ⊢ Iff (Ne x 0) (Ne x.orderTop Top.top)
  -/
  constructor
    /-
      case mp
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      x : HahnSeries Γ R
      ⊢ Ne x 0 → Ne x.orderTop Top.top
    -/
  · exact fun hx => Eq.mpr (congrArg (fun h ↦ h ≠ ⊤) (orderTop_of_ne hx)) WithTop.coe_ne_top
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      x : HahnSeries Γ R
      ⊢ Ne x.orderTop Top.top → Ne x 0
    -/
  · contrapose!
    /-
      case mpr
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      x : HahnSeries Γ R
      ⊢ Eq x 0 → Eq x.orderTop Top.top
    -/
    simp_all only [orderTop_zero, implies_true]
    /-
      🎉 no goals
    -/


theorem orderTop_eq_top_iff {x : HahnSeries Γ R} : orderTop x = ⊤ ↔ x = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    ⊢ Iff (Eq x.orderTop Top.top) (Eq x 0)
  -/
  constructor
    /-
      case mp
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      x : HahnSeries Γ R
      ⊢ Eq x.orderTop Top.top → Eq x 0
    -/
  · contrapose!
    /-
      case mp
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      x : HahnSeries Γ R
      ⊢ Ne x 0 → Ne x.orderTop Top.top
    -/
    exact ne_zero_iff_orderTop.mp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      x : HahnSeries Γ R
      ⊢ Eq x 0 → Eq x.orderTop Top.top
    -/
  · simp_all only [orderTop_zero, implies_true]
    /-
      🎉 no goals
    -/


theorem orderTop_eq_of_le {x : HahnSeries Γ R} {g : Γ} (hg : g ∈ x.support)
    (hx : ∀ g' ∈ x.support, g ≤ g') : orderTop x = g := by
  rw [orderTop_of_ne <| support_nonempty_iff.mp <| Set.nonempty_of_mem hg,
    x.isWF_support.min_eq_of_le hg hx]


theorem untop_orderTop_of_ne_zero {x : HahnSeries Γ R} (hx : x ≠ 0) :
    WithTop.untop x.orderTop (ne_zero_iff_orderTop.mp hx) =
      x.isWF_support.min (support_nonempty_iff.2 hx) :=
    WithTop.coe_inj.mp ((WithTop.coe_untop (orderTop x) (ne_zero_iff_orderTop.mp hx)).trans
      (orderTop_of_ne hx))


theorem coeff_orderTop_ne {x : HahnSeries Γ R} {g : Γ} (hg : x.orderTop = g) :
    x.coeff g ≠ 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    g : Γ
    hg : Eq x.orderTop ↑g
    ⊢ Ne (x.coeff g) 0
  -/
  have h : orderTop x ≠ ⊤ := by simp_all only [ne_eq, WithTop.coe_ne_top, not_false_eq_true]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    g : Γ
    hg : Eq x.orderTop ↑g
    h : Ne x.orderTop Top.top
    ⊢ Ne (x.coeff g) 0
  -/
  have hx : x ≠ 0 := ne_zero_iff_orderTop.mpr h
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    g : Γ
    hg : Eq x.orderTop ↑g
    h : Ne x.orderTop Top.top
    hx : Ne x 0
    ⊢ Ne (x.coeff g) 0
  -/
  rw [orderTop_of_ne hx, WithTop.coe_eq_coe] at hg
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    g : Γ
    h : Ne x.orderTop Top.top
    hx : Ne x 0
    hg : Eq (⋯.min ⋯) g
    ⊢ Ne (x.coeff g) 0
  -/
  rw [← hg]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    g : Γ
    h : Ne x.orderTop Top.top
    hx : Ne x 0
    hg : Eq (⋯.min ⋯) g
    ⊢ Ne (x.coeff (⋯.min ⋯)) 0
  -/
  exact x.isWF_support.min_mem (support_nonempty_iff.2 hx)
  /-
    🎉 no goals
  -/


theorem orderTop_le_of_coeff_ne_zero {Γ} [LinearOrder Γ] {x : HahnSeries Γ R}
    {g : Γ} (h : x.coeff g ≠ 0) : x.orderTop ≤ g := by
  /-
    R : Type u_3
    inst✝¹ : Zero R
    Γ : Type u_5
    inst✝ : LinearOrder Γ
    x : HahnSeries Γ R
    g : Γ
    h : Ne (x.coeff g) 0
    ⊢ LE.le x.orderTop ↑g
  -/
  rw [orderTop_of_ne (ne_zero_of_coeff_ne_zero h), WithTop.coe_le_coe]
  /-
    R : Type u_3
    inst✝¹ : Zero R
    Γ : Type u_5
    inst✝ : LinearOrder Γ
    x : HahnSeries Γ R
    g : Γ
    h : Ne (x.coeff g) 0
    ⊢ LE.le (⋯.min ⋯) g
  -/
  exact Set.IsWF.min_le _ _ ((mem_support _ _).2 h)
  /-
    🎉 no goals
  -/


@[simp]
theorem orderTop_single (h : r ≠ 0) : (single a r).orderTop = a :=
  (orderTop_of_ne (single_ne_zero h)).trans
    (WithTop.coe_inj.mpr (support_single_subset
      ((single a r).isWF_support.min_mem (support_nonempty_iff.2 (single_ne_zero h)))))


theorem orderTop_single_le : a ≤ (single a r).orderTop := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a : Γ
    r : R
    ⊢ LE.le (↑a) ((HahnSeries.single a) r).orderTop
  -/
  by_cases hr : r = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      a : Γ
      r : R
      hr : Eq r 0
      ⊢ LE.le (↑a) ((HahnSeries.single a) r).orderTop
    -/
  · simp only [hr, map_zero, orderTop_zero, le_top]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      a : Γ
      r : R
      hr : Not (Eq r 0)
      ⊢ LE.le (↑a) ((HahnSeries.single a) r).orderTop
    -/
  · rw [orderTop_single hr]
    /-
      🎉 no goals
    -/


theorem lt_orderTop_single {g g' : Γ} (hgg' : g < g') : g < (single g' r).orderTop :=
  lt_of_lt_of_le (WithTop.coe_lt_coe.mpr hgg') orderTop_single_le


theorem coeff_eq_zero_of_lt_orderTop {x : HahnSeries Γ R} {i : Γ} (hi : i < x.orderTop) :
    x.coeff i = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    i : Γ
    hi : LT.lt (↑i) x.orderTop
    ⊢ Eq (x.coeff i) 0
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case inl
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : PartialOrder Γ
      inst✝ : Zero R
      i : Γ
      hi : LT.lt (↑i) (HahnSeries.orderTop 0)
      ⊢ Eq (HahnSeries.coeff 0 i) 0
    -/
  · exact zero_coeff
    /-
      🎉 no goals
    -/
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    i : Γ
    hi : LT.lt (↑i) x.orderTop
    hx : Ne x 0
    ⊢ Eq (x.coeff i) 0
  -/
  contrapose! hi
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    i : Γ
    hx : Ne x 0
    hi : Ne (x.coeff i) 0
    ⊢ Not (LT.lt (↑i) x.orderTop)
  -/
  rw [← mem_support] at hi
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    i : Γ
    hx : Ne x 0
    hi : Membership.mem x.support i
    ⊢ Not (LT.lt (↑i) x.orderTop)
  -/
  rw [orderTop_of_ne hx, WithTop.coe_lt_coe]
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    i : Γ
    hx : Ne x 0
    hi : Membership.mem x.support i
    ⊢ Not (LT.lt i (⋯.min ⋯))
  -/
  exact Set.IsWF.not_lt_min _ _ hi
  /-
    🎉 no goals
  -/


open Classical in
/-- A leading coefficient of a Hahn series is the coefficient of a lowest-order nonzero term, or
zero if the series vanishes. -/
def leadingCoeff (x : HahnSeries Γ R) : R :=
  if h : x = 0 then 0 else x.coeff (x.isWF_support.min (support_nonempty_iff.2 h))


@[simp]
theorem leadingCoeff_zero : leadingCoeff (0 : HahnSeries Γ R) = 0 :=
  dif_pos rfl


theorem leadingCoeff_of_ne {x : HahnSeries Γ R} (hx : x ≠ 0) :
    x.leadingCoeff = x.coeff (x.isWF_support.min (support_nonempty_iff.2 hx)) :=
  dif_neg hx


theorem leadingCoeff_eq_iff {x : HahnSeries Γ R} : x.leadingCoeff = 0 ↔ x = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    ⊢ Iff (Eq x.leadingCoeff 0) (Eq x 0)
  -/
  refine { mp := ?_, mpr := fun hx => hx ▸ leadingCoeff_zero }
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    ⊢ Eq x.leadingCoeff 0 → Eq x 0
  -/
  contrapose!
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    x : HahnSeries Γ R
    ⊢ Ne x 0 → Ne x.leadingCoeff 0
  -/
  exact fun hx => (leadingCoeff_of_ne hx) ▸ coeff_orderTop_ne (orderTop_of_ne hx)
  /-
    🎉 no goals
  -/


theorem leadingCoeff_ne_iff {x : HahnSeries Γ R} : x.leadingCoeff ≠ 0 ↔ x ≠ 0 :=
  leadingCoeff_eq_iff.not


theorem leadingCoeff_of_single {a : Γ} {r : R} : leadingCoeff (single a r) = r := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a : Γ
    r : R
    ⊢ Eq ((HahnSeries.single a) r).leadingCoeff r
  -/
  simp only [leadingCoeff, single_eq_zero_iff]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : Zero R
    a : Γ
    r : R
    ⊢ Eq (dite (Eq ((HahnSeries.single a) r) 0) (fun h => 0) fun h => ((HahnSeries …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : r = 0 <;> simp [h]
                         /-
                           🎉 no goals
                         -/


open Classical in
/-- The order of a nonzero Hahn series `x` is a minimal element of `Γ` where `x` has a
  nonzero coefficient, the order of 0 is 0. -/
def order (x : HahnSeries Γ R) : Γ :=
  if h : x = 0 then 0 else x.isWF_support.min (support_nonempty_iff.2 h)


@[simp]
theorem order_zero : order (0 : HahnSeries Γ R) = 0 :=
  dif_pos rfl


theorem order_of_ne {x : HahnSeries Γ R} (hx : x ≠ 0) :
    order x = x.isWF_support.min (support_nonempty_iff.2 hx) :=
  dif_neg hx


theorem order_eq_orderTop_of_ne {x : HahnSeries Γ R} (hx : x ≠ 0) : order x = orderTop x := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    hx : Ne x 0
    ⊢ Eq (↑x.order) x.orderTop
  -/
  rw [order_of_ne hx, orderTop_of_ne hx]
  /-
    🎉 no goals
  -/


theorem coeff_order_ne_zero {x : HahnSeries Γ R} (hx : x ≠ 0) : x.coeff x.order ≠ 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    hx : Ne x 0
    ⊢ Ne (x.coeff x.order) 0
  -/
  rw [order_of_ne hx]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    hx : Ne x 0
    ⊢ Ne (x.coeff (⋯.min ⋯)) 0
  -/
  exact x.isWF_support.min_mem (support_nonempty_iff.2 hx)
  /-
    🎉 no goals
  -/


theorem order_le_of_coeff_ne_zero {Γ} [AddMonoid Γ] [LinearOrder Γ] {x : HahnSeries Γ R}
    {g : Γ} (h : x.coeff g ≠ 0) : x.order ≤ g :=
  le_trans (le_of_eq (order_of_ne (ne_zero_of_coeff_ne_zero h)))
    (Set.IsWF.min_le _ _ ((mem_support _ _).2 h))


@[simp]
theorem order_single (h : r ≠ 0) : (single a r).order = a :=
  (order_of_ne (single_ne_zero h)).trans
    (support_single_subset
      ((single a r).isWF_support.min_mem (support_nonempty_iff.2 (single_ne_zero h))))


theorem coeff_eq_zero_of_lt_order {x : HahnSeries Γ R} {i : Γ} (hi : i < x.order) :
    x.coeff i = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    i : Γ
    hi : LT.lt i x.order
    ⊢ Eq (x.coeff i) 0
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case inl
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      i : Γ
      hi : LT.lt i (HahnSeries.order 0)
      ⊢ Eq (HahnSeries.coeff 0 i) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    i : Γ
    hi : LT.lt i x.order
    hx : Ne x 0
    ⊢ Eq (x.coeff i) 0
  -/
  contrapose! hi
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    i : Γ
    hx : Ne x 0
    hi : Ne (x.coeff i) 0
    ⊢ Not (LT.lt i x.order)
  -/
  rw [← mem_support] at hi
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    i : Γ
    hx : Ne x 0
    hi : Membership.mem x.support i
    ⊢ Not (LT.lt i x.order)
  -/
  rw [order_of_ne hx]
  /-
    case inr
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    i : Γ
    hx : Ne x 0
    hi : Membership.mem x.support i
    ⊢ Not (LT.lt i (⋯.min ⋯))
  -/
  exact Set.IsWF.not_lt_min _ _ hi
  /-
    🎉 no goals
  -/


theorem zero_lt_orderTop_iff {x : HahnSeries Γ R} (hx : x ≠ 0) :
    0 < x.orderTop ↔ 0 < x.order := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    hx : Ne x 0
    ⊢ Iff (LT.lt 0 x.orderTop) (LT.lt 0 x.order)
  -/
  simp_all [orderTop_of_ne hx, order_of_ne hx]
  /-
    🎉 no goals
  -/


theorem zero_lt_orderTop_of_order {x : HahnSeries Γ R} (hx : 0 < x.order) : 0 < x.orderTop := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    hx : LT.lt 0 x.order
    ⊢ LT.lt 0 x.orderTop
  -/
  by_cases h : x = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      x : HahnSeries Γ R
      hx : LT.lt 0 x.order
      h : Eq x 0
      ⊢ LT.lt 0 x.orderTop
    -/
  · simp_all only [order_zero, lt_self_iff_false]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      x : HahnSeries Γ R
      hx : LT.lt 0 x.order
      h : Not (Eq x 0)
      ⊢ LT.lt 0 x.orderTop
    -/
  · exact (zero_lt_orderTop_iff h).mpr hx
    /-
      🎉 no goals
    -/


theorem zero_le_orderTop_iff {x : HahnSeries Γ R} : 0 ≤ x.orderTop ↔ 0 ≤ x.order := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    ⊢ Iff (LE.le 0 x.orderTop) (LE.le 0 x.order)
  -/
  by_cases h : x = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      x : HahnSeries Γ R
      h : Eq x 0
      ⊢ Iff (LE.le 0 x.orderTop) (LE.le 0 x.order)
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      x : HahnSeries Γ R
      h : Not (Eq x 0)
      ⊢ Iff (LE.le 0 x.orderTop) (LE.le 0 x.order)
    -/
  · simp_all [order_of_ne h, orderTop_of_ne h, zero_lt_orderTop_iff]
    /-
      🎉 no goals
    -/


theorem leadingCoeff_eq {x : HahnSeries Γ R} : x.leadingCoeff = x.coeff x.order := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : Zero Γ
    x : HahnSeries Γ R
    ⊢ Eq x.leadingCoeff (x.coeff x.order)
  -/
  by_cases h : x = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      x : HahnSeries Γ R
      h : Eq x 0
      ⊢ Eq x.leadingCoeff (x.coeff x.order)
    -/
  · rw [h, leadingCoeff_zero, zero_coeff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : Zero Γ
      x : HahnSeries Γ R
      h : Not (Eq x 0)
      ⊢ Eq x.leadingCoeff (x.coeff x.order)
    -/
  · rw [leadingCoeff_of_ne h, order_of_ne h]
    /-
      🎉 no goals
    -/


open Classical in
/-- Extends the domain of a `HahnSeries` by an `OrderEmbedding`. -/
def embDomain (f : Γ ↪o Γ') : HahnSeries Γ R → HahnSeries Γ' R := fun x =>
  { coeff := fun b : Γ' => if h : b ∈ f '' x.support then x.coeff (Classical.choose h) else 0
    isPWO_support' :=
      (x.isPWO_support.image_of_monotone f.monotone).mono fun b hb => by
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          S : Type u_4
          inst✝² : PartialOrder Γ
          inst✝¹ : Zero R
          a b✝ : Γ
          r : R
          inst✝ : PartialOrder Γ'
          f : OrderEmbedding Γ Γ'
          x : HahnSeries Γ R
          b : Γ'
          hb : Membership.mem (Function.support fun b => dite (Membership.mem (Set.image …
          ⊢ Membership.mem (Set.image (⇑f) x.support) b
        -/
        contrapose! hb
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          S : Type u_4
          inst✝² : PartialOrder Γ
          inst✝¹ : Zero R
          a b✝ : Γ
          r : R
          inst✝ : PartialOrder Γ'
          f : OrderEmbedding Γ Γ'
          x : HahnSeries Γ R
          b : Γ'
          hb : Not (Membership.mem (Set.image (⇑f) x.support) b)
          ⊢ Not (Membership.mem (Function.support fun b => dite (Membership.mem (Set.ima …
        -/
        rw [Function.mem_support, dif_neg hb, Classical.not_not] }
        /-
          🎉 no goals
        -/


@[simp]
theorem embDomain_coeff {f : Γ ↪o Γ'} {x : HahnSeries Γ R} {a : Γ} :
    (embDomain f x).coeff (f a) = x.coeff a := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x : HahnSeries Γ R
    a : Γ
    ⊢ Eq ((HahnSeries.embDomain f x).coeff (f a)) (x.coeff a)
  -/
  rw [embDomain]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x : HahnSeries Γ R
    a : Γ
    ⊢ Eq ({ coeff := fun b => dite (Membership.mem (Set.image (⇑f) x.support) b) ( …
  -/
  dsimp only
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x : HahnSeries Γ R
    a : Γ
    ⊢ Eq (dite (Membership.mem (Set.image (⇑f) x.support) (f a)) (fun h => x.coeff …
  -/
  by_cases ha : a ∈ x.support
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x : HahnSeries Γ R
      a : Γ
      ha : Membership.mem x.support a
      ⊢ Eq (dite (Membership.mem (Set.image (⇑f) x.support) (f a)) (fun h => x.coeff …
    -/
  · rw [dif_pos (Set.mem_image_of_mem f ha)]
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x : HahnSeries Γ R
      a : Γ
      ha : Membership.mem x.support a
      ⊢ Eq (x.coeff (Classical.choose ⋯)) (x.coeff a)
    -/
    exact congr rfl (f.injective (Classical.choose_spec (Set.mem_image_of_mem f ha)).2)
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x : HahnSeries Γ R
      a : Γ
      ha : Not (Membership.mem x.support a)
      ⊢ Eq (dite (Membership.mem (Set.image (⇑f) x.support) (f a)) (fun h => x.coeff …
    -/
  · rw [dif_neg, Classical.not_not.1 fun c => ha ((mem_support _ _).2 c)]
    /-
      case neg.hnc
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x : HahnSeries Γ R
      a : Γ
      ha : Not (Membership.mem x.support a)
      ⊢ Not (Membership.mem (Set.image (⇑f) x.support) (f a))
    -/
    contrapose! ha
    /-
      case neg.hnc
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x : HahnSeries Γ R
      a : Γ
      ha : Membership.mem (Set.image (⇑f) x.support) (f a)
      ⊢ Membership.mem x.support a
    -/
    obtain ⟨b, hb1, hb2⟩ := (Set.mem_image _ _ _).1 ha
    /-
      case neg.hnc.intro.intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x : HahnSeries Γ R
      a : Γ
      ha : Membership.mem (Set.image (⇑f) x.support) (f a)
      b : Γ
      hb1 : Membership.mem x.support b
      hb2 : Eq (f b) (f a)
      ⊢ Membership.mem x.support a
    -/
    rwa [f.injective hb2] at hb1
    /-
      🎉 no goals
    -/


@[simp]
theorem embDomain_mk_coeff {f : Γ → Γ'} (hfi : Function.Injective f)
    (hf : ∀ g g' : Γ, f g ≤ f g' ↔ g ≤ g') {x : HahnSeries Γ R} {a : Γ} :
    (embDomain ⟨⟨f, hfi⟩, hf _ _⟩ x).coeff (f a) = x.coeff a :=
  embDomain_coeff


theorem embDomain_notin_image_support {f : Γ ↪o Γ'} {x : HahnSeries Γ R} {b : Γ'}
    (hb : b ∉ f '' x.support) : (embDomain f x).coeff b = 0 :=
  dif_neg hb


theorem support_embDomain_subset {f : Γ ↪o Γ'} {x : HahnSeries Γ R} :
    support (embDomain f x) ⊆ f '' x.support := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x : HahnSeries Γ R
    ⊢ HasSubset.Subset (HahnSeries.embDomain f x).support (Set.image (⇑f) x.support)
  -/
  intro g hg
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x : HahnSeries Γ R
    g : Γ'
    hg : Membership.mem (HahnSeries.embDomain f x).support g
    ⊢ Membership.mem (Set.image (⇑f) x.support) g
  -/
  contrapose! hg
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x : HahnSeries Γ R
    g : Γ'
    hg : Not (Membership.mem (Set.image (⇑f) x.support) g)
    ⊢ Not (Membership.mem (HahnSeries.embDomain f x).support g)
  -/
  rw [mem_support, embDomain_notin_image_support hg, Classical.not_not]
  /-
    🎉 no goals
  -/


theorem embDomain_notin_range {f : Γ ↪o Γ'} {x : HahnSeries Γ R} {b : Γ'} (hb : b ∉ Set.range f) :
    (embDomain f x).coeff b = 0 :=
  embDomain_notin_image_support fun con => hb (Set.image_subset_range _ _ con)


@[simp]
theorem embDomain_zero {f : Γ ↪o Γ'} : embDomain f (0 : HahnSeries Γ R) = 0 := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    ⊢ Eq (HahnSeries.embDomain f 0) 0
  -/
  ext
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x✝ : Γ'
    ⊢ Eq ((HahnSeries.embDomain f 0).coeff x✝) (HahnSeries.coeff 0 x✝)
  -/
  simp [embDomain_notin_image_support]
  /-
    🎉 no goals
  -/


@[simp]
theorem embDomain_single {f : Γ ↪o Γ'} {g : Γ} {r : R} :
    embDomain f (single g r) = single (f g) r := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    g : Γ
    r : R
    ⊢ Eq (HahnSeries.embDomain f ((HahnSeries.single g) r)) ((HahnSeries.single (f …
  -/
  ext g'
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    g : Γ
    r : R
    g' : Γ'
    ⊢ Eq ((HahnSeries.embDomain f ((HahnSeries.single g) r)).coeff g') (((HahnSeri …
  -/
  by_cases h : g' = f g
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      g : Γ
      r : R
      g' : Γ'
      h : Eq g' (f g)
      ⊢ Eq ((HahnSeries.embDomain f ((HahnSeries.single g) r)).coeff g') (((HahnSeri …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    g : Γ
    r : R
    g' : Γ'
    h : Not (Eq g' (f g))
    ⊢ Eq ((HahnSeries.embDomain f ((HahnSeries.single g) r)).coeff g') (((HahnSeri …
  -/
  rw [embDomain_notin_image_support, single_coeff_of_ne h]
  /-
    case neg
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    g : Γ
    r : R
    g' : Γ'
    h : Not (Eq g' (f g))
    ⊢ Not (Membership.mem (Set.image (⇑f) ((HahnSeries.single g) r).support) g')
  -/
  by_cases hr : r = 0
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Zero R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      g : Γ
      r : R
      g' : Γ'
      h : Not (Eq g' (f g))
      hr : Eq r 0
      ⊢ Not (Membership.mem (Set.image (⇑f) ((HahnSeries.single g) r).support) g')
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    g : Γ
    r : R
    g' : Γ'
    h : Not (Eq g' (f g))
    hr : Not (Eq r 0)
    ⊢ Not (Membership.mem (Set.image (⇑f) ((HahnSeries.single g) r).support) g')
  -/
  rwa [support_single_of_ne hr, Set.image_singleton, Set.mem_singleton_iff]
  /-
    🎉 no goals
  -/


theorem embDomain_injective {f : Γ ↪o Γ'} :
    Function.Injective (embDomain f : HahnSeries Γ R → HahnSeries Γ' R) := fun x y xy => by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x y : HahnSeries Γ R
    xy : Eq (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)
    ⊢ Eq x y
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x y : HahnSeries Γ R
    xy : Eq (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)
    g : Γ
    ⊢ Eq (x.coeff g) (y.coeff g)
  -/
  rw [HahnSeries.ext_iff, funext_iff] at xy
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x y : HahnSeries Γ R
    xy : ∀ (x_1 : Γ'), Eq ((HahnSeries.embDomain f x).coeff x_1) ((HahnSeries.embD …
    g : Γ
    ⊢ Eq (x.coeff g) (y.coeff g)
  -/
  have xyg := xy (f g)
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Zero R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x y : HahnSeries Γ R
    xy : ∀ (x_1 : Γ'), Eq ((HahnSeries.embDomain f x).coeff x_1) ((HahnSeries.embD …
    g : Γ
    xyg : Eq ((HahnSeries.embDomain f x).coeff (f g)) ((HahnSeries.embDomain f y). …
    ⊢ Eq (x.coeff g) (y.coeff g)
  -/
  rwa [embDomain_coeff, embDomain_coeff] at xyg
  /-
    🎉 no goals
  -/


theorem forallLTEqZero_supp_BddBelow (f : Γ → R) (n : Γ) (hn : ∀(m : Γ), m < n → f m = 0) :
    BddBelow (Function.support f) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : Zero R
    inst✝ : LinearOrder Γ
    f : Γ → R
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    ⊢ BddBelow (Function.support f)
  -/
  simp only [BddBelow, Set.Nonempty, lowerBounds]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : Zero R
    inst✝ : LinearOrder Γ
    f : Γ → R
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    ⊢ Exists fun x => Membership.mem (setOf fun x => ∀ ⦃a : Γ⦄, Membership.mem (Fu …
  -/
  use n
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : Zero R
    inst✝ : LinearOrder Γ
    f : Γ → R
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    ⊢ Membership.mem (setOf fun x => ∀ ⦃a : Γ⦄, Membership.mem (Function.support f …
  -/
  intro m hm
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : Zero R
    inst✝ : LinearOrder Γ
    f : Γ → R
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    m : Γ
    hm : Membership.mem (Function.support f) m
    ⊢ LE.le n m
  -/
  rw [Function.mem_support, ne_eq] at hm
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : Zero R
    inst✝ : LinearOrder Γ
    f : Γ → R
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    m : Γ
    hm : Not (Eq (f m) 0)
    ⊢ LE.le n m
  -/
  exact not_lt.mp (mt (hn m) hm)
  /-
    🎉 no goals
  -/


theorem BddBelow_zero [Nonempty Γ] : BddBelow (Function.support (0 : Γ → R)) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : Zero R
    inst✝¹ : LinearOrder Γ
    inst✝ : Nonempty Γ
    ⊢ BddBelow (Function.support 0)
  -/
  simp only [support_zero', bddBelow_empty]
  /-
    🎉 no goals
  -/


theorem suppBddBelow_supp_PWO (f : Γ → R)
    (hf : BddBelow (Function.support f)) :
    (Function.support f).IsPWO :=
  Set.isWF_iff_isPWO.mp hf.wellFoundedOn_lt


/-- Construct a Hahn series from any function whose support is bounded below. -/
@[simps]
def ofSuppBddBelow (f : Γ → R) (hf : BddBelow (Function.support f)) : HahnSeries Γ R where
  coeff := f
  isPWO_support' := suppBddBelow_supp_PWO f hf


@[simp]
theorem zero_ofSuppBddBelow [Nonempty Γ] : ofSuppBddBelow 0 BddBelow_zero = (0 : HahnSeries Γ R) :=
  rfl


theorem order_ofForallLtEqZero [Zero Γ] (f : Γ → R) (hf : f ≠ 0) (n : Γ)
    (hn : ∀(m : Γ), m < n → f m = 0) :
    n ≤ order (ofSuppBddBelow f (forallLTEqZero_supp_BddBelow f n hn)) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    ⊢ LE.le n (HahnSeries.ofSuppBddBelow f ⋯).order
  -/
  dsimp only [order]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    ⊢ LE.le n (dite (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0) (fun h => 0) fun h => ⋯ …
  -/
  by_cases h : ofSuppBddBelow f (forallLTEqZero_supp_BddBelow f n hn) = 0
  /-
    case pos
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    h : Eq (HahnSeries.ofSuppBddBelow f ⋯) 0
    ⊢ LE.le n (dite (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0) (fun h => 0) fun h => ⋯ …
  -/
  cases h
    /-
      case pos.refl
      Γ : Type u_1
      R : Type u_3
      inst✝³ : Zero R
      inst✝² : LinearOrder Γ
      inst✝¹ : LocallyFiniteOrder Γ
      inst✝ : Zero Γ
      n : Γ
      hf : Ne 0 0
      hn : ∀ (m : Γ), LT.lt m n → Eq (0 m) 0
      ⊢ LE.le n (dite (Eq (HahnSeries.ofSuppBddBelow 0 ⋯) 0) (fun h => 0) fun h => ⋯ …
    -/
  · exact (hf rfl).elim
    /-
      🎉 no goals
    -/
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    h : Not (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0)
    ⊢ LE.le n (dite (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0) (fun h => 0) fun h => ⋯ …
  -/
  simp_all only [dite_false]
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    h : Not (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0)
    ⊢ LE.le n (⋯.min ⋯)
  -/
  rw [Set.IsWF.le_min_iff]
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    h : Not (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0)
    ⊢ ∀ (b : Γ), Membership.mem (HahnSeries.ofSuppBddBelow f ⋯).support b → LE.le  …
  -/
  intro m hm
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    h : Not (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0)
    m : Γ
    hm : Membership.mem (HahnSeries.ofSuppBddBelow f ⋯).support m
    ⊢ LE.le n m
  -/
  rw [HahnSeries.support, Function.mem_support, ne_eq] at hm
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero R
    inst✝² : LinearOrder Γ
    inst✝¹ : LocallyFiniteOrder Γ
    inst✝ : Zero Γ
    f : Γ → R
    hf : Ne f 0
    n : Γ
    hn : ∀ (m : Γ), LT.lt m n → Eq (f m) 0
    h : Not (Eq (HahnSeries.ofSuppBddBelow f ⋯) 0)
    m : Γ
    hm : Not (Eq ((HahnSeries.ofSuppBddBelow f ⋯).coeff m) 0)
    ⊢ LE.le n m
  -/
  exact not_lt.mp (mt (hn m) hm)
  /-
    🎉 no goals
  -/


