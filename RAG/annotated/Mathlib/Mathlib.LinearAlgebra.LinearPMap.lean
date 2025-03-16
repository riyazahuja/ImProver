/-- A `LinearPMap R E F` or `E →ₗ.[R] F` is a linear map from a submodule of `E` to `F`. -/
structure LinearPMap (R : Type u) [Ring R] (E : Type v) [AddCommGroup E] [Module R E] (F : Type w)
  [AddCommGroup F] [Module R F] where
  domain : Submodule R E
  toFun : domain →ₗ[R] F


@[inherit_doc] notation:25 E " →ₗ.[" R:25 "] " F:0 => LinearPMap R E F


@[coe]
def toFun' (f : E →ₗ.[R] F) : f.domain → F := f.toFun


instance : CoeFun (E →ₗ.[R] F) fun f : E →ₗ.[R] F => f.domain → F :=
  ⟨toFun'⟩


@[simp]
theorem toFun_eq_coe (f : E →ₗ.[R] F) (x : f.domain) : f.toFun x = f x :=
  rfl


@[ext (iff := false)]
theorem ext {f g : E →ₗ.[R] F} (h : f.domain = g.domain)
    (h' : ∀ ⦃x : f.domain⦄ ⦃y : g.domain⦄ (_h : (x : E) = y), f x = g y) : f = g := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : Eq f.domain g.domain
    h' : ∀ ⦃x : Subtype fun x => Membership.mem f.domain x⦄ ⦃y : Subtype fun x =>  …
    ⊢ Eq f g
  -/
  rcases f with ⟨f_dom, f⟩
  /-
    case mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : LinearPMap R E F
    f_dom : Submodule R E
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem f_dom x) F
    h : Eq { domain := f_dom, toFun := f }.domain g.domain
    h' : ∀ ⦃x : Subtype fun x => Membership.mem { domain := f_dom, toFun := f }.do …
    ⊢ Eq { domain := f_dom, toFun := f } g
  -/
  rcases g with ⟨g_dom, g⟩
  /-
    case mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f_dom : Submodule R E
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem f_dom x) F
    g_dom : Submodule R E
    g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem g_dom x) F
    h : Eq { domain := f_dom, toFun := f }.domain { domain := g_dom, toFun := g }. …
    h' : ∀ ⦃x : Subtype fun x => Membership.mem { domain := f_dom, toFun := f }.do …
    ⊢ Eq { domain := f_dom, toFun := f } { domain := g_dom, toFun := g }
  -/
  obtain rfl : f_dom = g_dom := h
  /-
    case mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f_dom : Submodule R E
    f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem f_dom x) F
    h' : ∀ ⦃x : Subtype fun x => Membership.mem { domain := f_dom, toFun := f }.do …
    ⊢ Eq { domain := f_dom, toFun := f } { domain := f_dom, toFun := g }
  -/
  obtain rfl : f = g := LinearMap.ext fun x => h' rfl
  /-
    case mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f_dom : Submodule R E
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem f_dom x) F
    h' : ∀ ⦃x y : Subtype fun x => Membership.mem { domain := f_dom, toFun := f }. …
    ⊢ Eq { domain := f_dom, toFun := f } { domain := f_dom, toFun := f }
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_zero (f : E →ₗ.[R] F) : f 0 = 0 :=
  f.toFun.map_zero


theorem ext_iff {f g : E →ₗ.[R] F} :
    f = g ↔
      ∃ _domain_eq : f.domain = g.domain,
        ∀ ⦃x : f.domain⦄ ⦃y : g.domain⦄ (_h : (x : E) = y), f x = g y :=
  ⟨fun EQ =>
    EQ ▸
      ⟨rfl, fun x y h => by
        /-
          R : Type u_1
          inst✝⁴ : Ring R
          E : Type u_2
          inst✝³ : AddCommGroup E
          inst✝² : Module R E
          F : Type u_3
          inst✝¹ : AddCommGroup F
          inst✝ : Module R F
          f g : LinearPMap R E F
          EQ : Eq f g
          x y : Subtype fun x => Membership.mem f.domain x
          h : Eq ↑x ↑y
          ⊢ Eq (↑f x) (↑f y)
        -/
        congr
        /-
          case e_a
          R : Type u_1
          inst✝⁴ : Ring R
          E : Type u_2
          inst✝³ : AddCommGroup E
          inst✝² : Module R E
          F : Type u_3
          inst✝¹ : AddCommGroup F
          inst✝ : Module R F
          f g : LinearPMap R E F
          EQ : Eq f g
          x y : Subtype fun x => Membership.mem f.domain x
          h : Eq ↑x ↑y
          ⊢ Eq x y
        -/
        exact mod_cast h⟩,
        /-
          🎉 no goals
        -/
    fun ⟨deq, feq⟩ => ext deq feq⟩


theorem ext' {s : Submodule R E} {f g : s →ₗ[R] F} (h : f = g) : mk s f = mk s g :=
  h ▸ rfl


theorem map_add (f : E →ₗ.[R] F) (x y : f.domain) : f (x + y) = f x + f y :=
  f.toFun.map_add x y


theorem map_neg (f : E →ₗ.[R] F) (x : f.domain) : f (-x) = -f x :=
  f.toFun.map_neg x


theorem map_sub (f : E →ₗ.[R] F) (x y : f.domain) : f (x - y) = f x - f y :=
  f.toFun.map_sub x y


theorem map_smul (f : E →ₗ.[R] F) (c : R) (x : f.domain) : f (c • x) = c • f x :=
  f.toFun.map_smul c x


@[simp]
theorem mk_apply (p : Submodule R E) (f : p →ₗ[R] F) (x : p) : mk p f x = f x :=
  rfl


/-- The unique `LinearPMap` on `R ∙ x` that sends `x` to `y`. This version works for modules
over rings, and requires a proof of `∀ c, c • x = 0 → c • y = 0`. -/
noncomputable def mkSpanSingleton' (x : E) (y : F) (H : ∀ c : R, c • x = 0 → c • y = 0) :
    E →ₗ.[R] F where
  domain := R ∙ x
  toFun :=
    have H : ∀ c₁ c₂ : R, c₁ • x = c₂ • x → c₁ • y = c₂ • y := by
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        x : E
        y : F
        H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
        ⊢ ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul c₁ …
      -/
      intro c₁ c₂ h
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        x : E
        y : F
        H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
        c₁ c₂ : R
        h : Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x)
        ⊢ Eq (HSMul.hSMul c₁ y) (HSMul.hSMul c₂ y)
      -/
      rw [← sub_eq_zero, ← sub_smul] at h ⊢
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        x : E
        y : F
        H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
        c₁ c₂ : R
        h : Eq (HSMul.hSMul (HSub.hSub c₁ c₂) x) 0
        ⊢ Eq (HSMul.hSMul (HSub.hSub c₁ c₂) y) 0
      -/
      exact H _ h
      /-
        🎉 no goals
      -/
    { toFun := fun z => Classical.choose (mem_span_singleton.1 z.prop) • y
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      -- Porting note: Were `Classical.choose_spec (mem_span_singleton.1 _)`.
      map_add' := fun y z => by
        /-
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y✝ : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y✝) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          y z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton …
          ⊢ Eq ((fun z => HSMul.hSMul (Classical.choose ⋯) y✝) (HAdd.hAdd y z)) (HAdd.hA …
        -/
        beta_reduce
        /-
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y✝ : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y✝) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          y z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton …
          ⊢ Eq (HSMul.hSMul (Classical.choose ⋯) y✝) (HAdd.hAdd (HSMul.hSMul (Classical. …
        -/
        rw [← add_smul]
        /-
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y✝ : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y✝) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          y z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton …
          ⊢ Eq (HSMul.hSMul (Classical.choose ⋯) y✝) (HSMul.hSMul (HAdd.hAdd (Classical. …
        -/
        apply H
        simp only [add_smul, sub_smul,
          fun w : R ∙ x => Classical.choose_spec (mem_span_singleton.1 w.prop)]
        /-
          case a
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y✝ : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y✝) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          y z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton …
          ⊢ Eq (↑(HAdd.hAdd y z)) (HAdd.hAdd ↑y ↑z)
        -/
        apply coe_add
        /-
          🎉 no goals
        -/
      map_smul' := fun c z => by
        /-
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          c : R
          z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton x …
          ⊢ Eq ({ toFun := fun z => HSMul.hSMul (Classical.choose ⋯) y, map_add' := ⋯ }. …
        -/
        beta_reduce
        /-
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          c : R
          z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton x …
          ⊢ Eq ({ toFun := fun z => HSMul.hSMul (Classical.choose ⋯) y, map_add' := ⋯ }. …
        -/
        rw [smul_smul]
        /-
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          c : R
          z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton x …
          ⊢ Eq ({ toFun := fun z => HSMul.hSMul (Classical.choose ⋯) y, map_add' := ⋯ }. …
        -/
        apply H
        simp only [mul_smul,
          fun w : R ∙ x => Classical.choose_spec (mem_span_singleton.1 w.prop)]
        /-
          case a
          R : Type u_1
          inst✝⁶ : Ring R
          E : Type u_2
          inst✝⁵ : AddCommGroup E
          inst✝⁴ : Module R E
          F : Type u_3
          inst✝³ : AddCommGroup F
          inst✝² : Module R F
          G : Type u_4
          inst✝¹ : AddCommGroup G
          inst✝ : Module R G
          x : E
          y : F
          H✝ : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
          H : ∀ (c₁ c₂ : R), Eq (HSMul.hSMul c₁ x) (HSMul.hSMul c₂ x) → Eq (HSMul.hSMul  …
          c : R
          z : Subtype fun x_1 => Membership.mem (Submodule.span R (Singleton.singleton x …
          ⊢ Eq (↑(HSMul.hSMul c z)) (HSMul.hSMul ((RingHom.id R) c) ↑z)
        -/
        apply coe_smul }
        /-
          🎉 no goals
        -/


@[simp]
theorem domain_mkSpanSingleton (x : E) (y : F) (H : ∀ c : R, c • x = 0 → c • y = 0) :
    (mkSpanSingleton' x y H).domain = R ∙ x :=
  rfl


@[simp]
theorem mkSpanSingleton'_apply (x : E) (y : F) (H : ∀ c : R, c • x = 0 → c • y = 0) (c : R) (h) :
    mkSpanSingleton' x y H ⟨c • x, h⟩ = c • y := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    c : R
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain (HSMul.hSMul c x)
    ⊢ Eq (↑(LinearPMap.mkSpanSingleton' x y H) ⟨HSMul.hSMul c x, h⟩) (HSMul.hSMul  …
  -/
  dsimp [mkSpanSingleton']
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    c : R
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain (HSMul.hSMul c x)
    ⊢ Eq (HSMul.hSMul (Classical.choose ⋯) y) (HSMul.hSMul c y)
  -/
  rw [← sub_eq_zero, ← sub_smul]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    c : R
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain (HSMul.hSMul c x)
    ⊢ Eq (HSMul.hSMul (HSub.hSub (Classical.choose ⋯) c) y) 0
  -/
  apply H
  /-
    case a
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    c : R
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain (HSMul.hSMul c x)
    ⊢ Eq (HSMul.hSMul (HSub.hSub (Classical.choose ⋯) c) x) 0
  -/
  simp only [sub_smul, one_smul, sub_eq_zero]
  /-
    case a
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    c : R
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain (HSMul.hSMul c x)
    ⊢ Eq (HSMul.hSMul (Classical.choose ⋯) x) (HSMul.hSMul c x)
  -/
  apply Classical.choose_spec (mem_span_singleton.1 h)
  /-
    🎉 no goals
  -/


@[simp]
theorem mkSpanSingleton'_apply_self (x : E) (y : F) (H : ∀ c : R, c • x = 0 → c • y = 0) (h) :
    mkSpanSingleton' x y H ⟨x, h⟩ = y := by
  -- Porting note: A placeholder should be specified before `convert`.
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain x
    ⊢ Eq (↑(LinearPMap.mkSpanSingleton' x y H) ⟨x, h⟩) y
  -/
  have := by refine mkSpanSingleton'_apply x y H 1 ?_; rwa [one_smul]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    x : E
    y : F
    H : ∀ (c : R), Eq (HSMul.hSMul c x) 0 → Eq (HSMul.hSMul c y) 0
    h : Membership.mem (LinearPMap.mkSpanSingleton' x y H).domain x
    this : Eq (↑(LinearPMap.mkSpanSingleton' x y H) ⟨HSMul.hSMul 1 x, ⋯⟩) (HSMul.h …
    ⊢ Eq (↑(LinearPMap.mkSpanSingleton' x y H) ⟨x, h⟩) y
  -/
                   /-
                     🎉 no goals
                   -/
  convert this <;> rw [one_smul]
                   /-
                     🎉 no goals
                   -/


/-- The unique `LinearPMap` on `span R {x}` that sends a non-zero vector `x` to `y`.
This version works for modules over division rings. -/
noncomputable abbrev mkSpanSingleton {K E F : Type*} [DivisionRing K] [AddCommGroup E] [Module K E]
    [AddCommGroup F] [Module K F] (x : E) (y : F) (hx : x ≠ 0) : E →ₗ.[K] F :=
  mkSpanSingleton' x y fun c hc =>
                                           /-
                                             R : Type u_1
                                             inst✝¹¹ : Ring R
                                             E✝ : Type u_2
                                             inst✝¹⁰ : AddCommGroup E✝
                                             inst✝⁹ : Module R E✝
                                             F✝ : Type u_3
                                             inst✝⁸ : AddCommGroup F✝
                                             inst✝⁷ : Module R F✝
                                             G : Type u_4
                                             inst✝⁶ : AddCommGroup G
                                             inst✝⁵ : Module R G
                                             K : Type u_5
                                             E : Type u_6
                                             F : Type u_7
                                             inst✝⁴ : DivisionRing K
                                             inst✝³ : AddCommGroup E
                                             inst✝² : Module K E
                                             inst✝¹ : AddCommGroup F
                                             inst✝ : Module K F
                                             x : E
                                             y : F
                                             hx : Ne x 0
                                             c : K
                                             hc✝ : Eq (HSMul.hSMul c x) 0
                                             hc : Eq c 0
                                             ⊢ Eq (HSMul.hSMul c y) 0
                                           -/
    (smul_eq_zero.1 hc).elim (fun hc => by rw [hc, zero_smul]) fun hx' => absurd hx' hx
                                           /-
                                             🎉 no goals
                                           -/


theorem mkSpanSingleton_apply (K : Type*) {E F : Type*} [DivisionRing K] [AddCommGroup E]
    [Module K E] [AddCommGroup F] [Module K F] {x : E} (hx : x ≠ 0) (y : F) :
    mkSpanSingleton x y hx ⟨x, (Submodule.mem_span_singleton_self x : x ∈ Submodule.span K {x})⟩ =
      y :=
  LinearPMap.mkSpanSingleton'_apply_self _ _ _ _


/-- Projection to the first coordinate as a `LinearPMap` -/
protected def fst (p : Submodule R E) (p' : Submodule R F) : E × F →ₗ.[R] E where
  domain := p.prod p'
  toFun := (LinearMap.fst R E F).comp (p.prod p').subtype


@[simp]
theorem fst_apply (p : Submodule R E) (p' : Submodule R F) (x : p.prod p') :
    LinearPMap.fst p p' x = (x : E × F).1 :=
  rfl


/-- Projection to the second coordinate as a `LinearPMap` -/
protected def snd (p : Submodule R E) (p' : Submodule R F) : E × F →ₗ.[R] F where
  domain := p.prod p'
  toFun := (LinearMap.snd R E F).comp (p.prod p').subtype


@[simp]
theorem snd_apply (p : Submodule R E) (p' : Submodule R F) (x : p.prod p') :
    LinearPMap.snd p p' x = (x : E × F).2 :=
  rfl


instance le : LE (E →ₗ.[R] F) :=
  ⟨fun f g => f.domain ≤ g.domain ∧ ∀ ⦃x : f.domain⦄ ⦃y : g.domain⦄ (_h : (x : E) = y), f x = g y⟩


theorem apply_comp_inclusion {T S : E →ₗ.[R] F} (h : T ≤ S) (x : T.domain) :
    T x = S (Submodule.inclusion h.1 x) :=
  h.2 rfl


theorem exists_of_le {T S : E →ₗ.[R] F} (h : T ≤ S) (x : T.domain) :
    ∃ y : S.domain, (x : E) = y ∧ T x = S y :=
  ⟨⟨x.1, h.1 x.2⟩, ⟨rfl, h.2 rfl⟩⟩


theorem eq_of_le_of_domain_eq {f g : E →ₗ.[R] F} (hle : f ≤ g) (heq : f.domain = g.domain) :
    f = g :=
  ext heq hle.2


/-- Given two partial linear maps `f`, `g`, the set of points `x` such that
both `f` and `g` are defined at `x` and `f x = g x` form a submodule. -/
def eqLocus (f g : E →ₗ.[R] F) : Submodule R E where
  carrier := { x | ∃ (hf : x ∈ f.domain) (hg : x ∈ g.domain), f ⟨x, hf⟩ = g ⟨x, hg⟩ }
  zero_mem' := ⟨zero_mem _, zero_mem _, f.map_zero.trans g.map_zero.symm⟩
  add_mem' := fun {x y} ⟨hfx, hgx, hx⟩ ⟨hfy, hgy, hy⟩ =>
    ⟨add_mem hfx hfy, add_mem hgx hgy, by
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        x y : E
        x✝¹ : Membership.mem (setOf fun x => Exists fun hf => Exists fun hg => Eq (↑f  …
        x✝ : Membership.mem (setOf fun x => Exists fun hf => Exists fun hg => Eq (↑f ⟨ …
        hfx : Membership.mem f.domain x
        hgx : Membership.mem g.domain x
        hx : Eq (↑f ⟨x, hfx⟩) (↑g ⟨x, hgx⟩)
        hfy : Membership.mem f.domain y
        hgy : Membership.mem g.domain y
        hy : Eq (↑f ⟨y, hfy⟩) (↑g ⟨y, hgy⟩)
        ⊢ Eq (↑f ⟨HAdd.hAdd x y, ⋯⟩) (↑g ⟨HAdd.hAdd x y, ⋯⟩)
      -/
      erw [f.map_add ⟨x, hfx⟩ ⟨y, hfy⟩, g.map_add ⟨x, hgx⟩ ⟨y, hgy⟩, hx, hy]⟩
      /-
        🎉 no goals
      -/
  -- Porting note: `by rintro` is required, or error of a free variable happens.
  smul_mem' := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      ⊢ ∀ (c : R) {x : E}, Membership.mem { carrier := setOf fun x => Exists fun hf  …
    -/
    rintro c x ⟨hfx, hgx, hx⟩
    exact
      ⟨smul_mem _ c hfx, smul_mem _ c hgx,
        by erw [f.map_smul c ⟨x, hfx⟩, g.map_smul c ⟨x, hgx⟩, hx]⟩


instance bot : Bot (E →ₗ.[R] F) :=
  ⟨⟨⊥, 0⟩⟩


instance inhabited : Inhabited (E →ₗ.[R] F) :=
  ⟨⊥⟩


instance semilatticeInf : SemilatticeInf (E →ₗ.[R] F) where
  le := (· ≤ ·)
  le_refl f := ⟨le_refl f.domain, fun _ _ h => Subtype.eq h ▸ rfl⟩
  le_trans := fun _ _ _ ⟨fg_le, fg_eq⟩ ⟨gh_le, gh_eq⟩ =>
    ⟨le_trans fg_le gh_le, fun x _ hxz =>
      have hxy : (x : E) = inclusion fg_le x := rfl
      (fg_eq hxy).trans (gh_eq <| hxy.symm.trans hxz)⟩
  le_antisymm _ _ fg gf := eq_of_le_of_domain_eq fg (le_antisymm fg.1 gf.1)
  inf f g := ⟨f.eqLocus g, f.toFun.comp <| inclusion fun _x hx => hx.fst⟩
  le_inf := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      ⊢ ∀ (a b c : LinearPMap R E F), LE.le a b → LE.le a c → LE.le a ((fun f g => { …
    -/
    intro f g h ⟨fg_le, fg_eq⟩ ⟨fh_le, fh_eq⟩
    exact ⟨fun x hx =>
      ⟨fg_le hx, fh_le hx, by
        -- Porting note: `[exact ⟨x, hx⟩, rfl, rfl]` → `[skip, exact ⟨x, hx⟩, skip] <;> rfl`
        convert (fg_eq _).symm.trans (fh_eq _) <;> [skip; exact ⟨x, hx⟩; skip] <;> rfl⟩,
      fun x ⟨y, yg, hy⟩ h => by
        apply fg_eq
        exact h⟩
  inf_le_left f _ := ⟨fun _ hx => hx.fst, fun _ _ h => congr_arg f <| Subtype.eq <| h⟩
  inf_le_right _ g :=
    ⟨fun _ hx => hx.snd.fst, fun ⟨_, _, _, hx⟩ _ h => hx.trans <| congr_arg g <| Subtype.eq <| h⟩


instance orderBot : OrderBot (E →ₗ.[R] F) where
  bot := ⊥
  bot_le f :=
    ⟨bot_le, fun x y h => by
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem Bot.bot.domain x
        y : Subtype fun x => Membership.mem f.domain x
        h : Eq ↑x ↑y
        ⊢ Eq (↑Bot.bot x) (↑f y)
      -/
      have hx : x = 0 := Subtype.eq ((mem_bot R).1 x.2)
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem Bot.bot.domain x
        y : Subtype fun x => Membership.mem f.domain x
        h : Eq ↑x ↑y
        hx : Eq x 0
        ⊢ Eq (↑Bot.bot x) (↑f y)
      -/
      have hy : y = 0 := Subtype.eq (h.symm.trans (congr_arg _ hx))
      /-
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem Bot.bot.domain x
        y : Subtype fun x => Membership.mem f.domain x
        h : Eq ↑x ↑y
        hx : Eq x 0
        hy : Eq y 0
        ⊢ Eq (↑Bot.bot x) (↑f y)
      -/
      rw [hx, hy, map_zero, map_zero]⟩
      /-
        🎉 no goals
      -/


theorem le_of_eqLocus_ge {f g : E →ₗ.[R] F} (H : f.domain ≤ f.eqLocus g) : f ≤ g :=
  suffices f ≤ f ⊓ g from le_trans this inf_le_right
  ⟨H, fun _x _y hxy => ((inf_le_left : f ⊓ g ≤ f).2 hxy.symm).symm⟩


theorem domain_mono : StrictMono (@domain R _ E _ _ F _ _) := fun _f _g hlt =>
  lt_of_le_of_ne hlt.1.1 fun heq => ne_of_lt hlt <| eq_of_le_of_domain_eq (le_of_lt hlt) heq


private theorem sup_aux (f g : E →ₗ.[R] F)
    (h : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y) :
    ∃ fg : ↥(f.domain ⊔ g.domain) →ₗ[R] F,
      ∀ (x : f.domain) (y : g.domain) (z : ↥(f.domain ⊔ g.domain)),
        (x : E) + y = ↑z → fg z = f x + g y := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    ⊢ Exists fun fg => ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Sub …
  -/
  choose x hx y hy hxy using fun z : ↥(f.domain ⊔ g.domain) => mem_sup.1 z.prop
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
    hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
    y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
    hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
    hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
    ⊢ Exists fun fg => ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Sub …
  -/
  set fg := fun z => f ⟨x z, hx z⟩ + g ⟨y z, hy z⟩
  have fg_eq : ∀ (x' : f.domain) (y' : g.domain) (z' : ↥(f.domain ⊔ g.domain))
      (_H : (x' : E) + y' = z'), fg z' = f x' + g y' := by
    intro x' y' z' H
    dsimp [fg]
    rw [add_comm, ← sub_eq_sub_iff_add_eq_add, eq_comm, ← map_sub, ← map_sub]
    apply h
    simp only [← eq_sub_iff_add_eq] at hxy
    simp only [AddSubgroupClass.coe_sub, coe_mk, coe_mk, hxy, ← sub_add, ← sub_sub, sub_self,
      zero_sub, ← H]
    apply neg_add_eq_sub
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
    hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
    y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
    hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
    hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
    fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
    fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
    ⊢ Exists fun fg => ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Sub …
  -/
  use { toFun := fg, map_add' := ?_, map_smul' := ?_ }, fg_eq
    /-
      case w.refine_1
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      ⊢ ∀ (x y : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Eq  …
    -/
  · rintro ⟨z₁, hz₁⟩ ⟨z₂, hz₂⟩
    /-
      case w.refine_1.mk.mk
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      z₁ : E
      hz₁ : Membership.mem (Max.max f.domain g.domain) z₁
      z₂ : E
      hz₂ : Membership.mem (Max.max f.domain g.domain) z₂
      ⊢ Eq (fg (HAdd.hAdd ⟨z₁, hz₁⟩ ⟨z₂, hz₂⟩)) (HAdd.hAdd (fg ⟨z₁, hz₁⟩) (fg ⟨z₂, h …
    -/
    rw [← add_assoc, add_right_comm (f _), ← map_add, add_assoc, ← map_add]
    /-
      case w.refine_1.mk.mk
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      z₁ : E
      hz₁ : Membership.mem (Max.max f.domain g.domain) z₁
      z₂ : E
      hz₂ : Membership.mem (Max.max f.domain g.domain) z₂
      ⊢ Eq (fg (HAdd.hAdd ⟨z₁, hz₁⟩ ⟨z₂, hz₂⟩)) (HAdd.hAdd (↑f (HAdd.hAdd ⟨x ⟨z₁, hz …
    -/
    apply fg_eq
    /-
      case w.refine_1.mk.mk._H
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      z₁ : E
      hz₁ : Membership.mem (Max.max f.domain g.domain) z₁
      z₂ : E
      hz₂ : Membership.mem (Max.max f.domain g.domain) z₂
      ⊢ Eq (HAdd.hAdd ↑(HAdd.hAdd ⟨x ⟨z₁, hz₁⟩, ⋯⟩ ⟨x ⟨z₂, hz₂⟩, ⋯⟩) ↑(HAdd.hAdd ⟨y  …
    -/
    simp only [coe_add, coe_mk, ← add_assoc]
    /-
      case w.refine_1.mk.mk._H
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      z₁ : E
      hz₁ : Membership.mem (Max.max f.domain g.domain) z₁
      z₂ : E
      hz₂ : Membership.mem (Max.max f.domain g.domain) z₂
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (x ⟨z₁, hz₁⟩) (x ⟨z₂, hz₂⟩)) (y ⟨z₁, hz₁ …
    -/
    rw [add_right_comm (x _), hxy, add_assoc, hxy, coe_mk, coe_mk]
    /-
      🎉 no goals
    -/
    /-
      case w.refine_2
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      ⊢ ∀ (m : R) (x_1 : Subtype fun x => Membership.mem (Max.max f.domain g.domain) …
    -/
  · intro c z
    /-
      case w.refine_2
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      c : R
      z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x
      ⊢ Eq ({ toFun := fg, map_add' := ⋯ }.toFun (HSMul.hSMul c z)) (HSMul.hSMul ((R …
    -/
    rw [smul_add, ← map_smul, ← map_smul]
    /-
      case w.refine_2
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      c : R
      z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x
      ⊢ Eq ({ toFun := fg, map_add' := ⋯ }.toFun (HSMul.hSMul c z)) (HAdd.hAdd (↑f ( …
    -/
    apply fg_eq
    /-
      case w.refine_2._H
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
      x : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hx : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      y : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → E
      hy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), Me …
      hxy : ∀ (z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x), E …
      fg : (Subtype fun x => Membership.mem (Max.max f.domain g.domain) x) → F := fu …
      fg_eq : ∀ (x' : Subtype fun x => Membership.mem f.domain x) (y' : Subtype fun  …
      c : R
      z : Subtype fun x => Membership.mem (Max.max f.domain g.domain) x
      ⊢ Eq (HAdd.hAdd ↑(HSMul.hSMul ((RingHom.id R) c) ⟨x z, ⋯⟩) ↑(HSMul.hSMul ((Rin …
    -/
    simp only [coe_smul, coe_mk, ← smul_add, hxy, RingHom.id_apply]
    /-
      🎉 no goals
    -/


/-- Given two partial linear maps that agree on the intersection of their domains,
`f.sup g h` is the unique partial linear map on `f.domain ⊔ g.domain` that agrees
with `f` and `g`. -/
protected noncomputable def sup (f g : E →ₗ.[R] F)
    (h : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y) : E →ₗ.[R] F :=
  ⟨_, Classical.choose (sup_aux f g h)⟩


@[simp]
theorem domain_sup (f g : E →ₗ.[R] F)
    (h : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y) :
    (f.sup g h).domain = f.domain ⊔ g.domain :=
  rfl


theorem sup_apply {f g : E →ₗ.[R] F} (H : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y)
    (x : f.domain) (y : g.domain) (z : ↥(f.domain ⊔ g.domain)) (hz : (↑x : E) + ↑y = ↑z) :
    f.sup g H z = f x + g y :=
  Classical.choose_spec (sup_aux f g H) x y z hz


protected theorem left_le_sup (f g : E →ₗ.[R] F)
    (h : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y) : f ≤ f.sup g h := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    ⊢ LE.le f (f.sup g h)
  -/
  refine ⟨le_sup_left, fun z₁ z₂ hz => ?_⟩
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    z₁ : Subtype fun x => Membership.mem f.domain x
    z₂ : Subtype fun x => Membership.mem (f.sup g h).domain x
    hz : Eq ↑z₁ ↑z₂
    ⊢ Eq (↑f z₁) (↑(f.sup g h) z₂)
  -/
  rw [← add_zero (f _), ← g.map_zero]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    z₁ : Subtype fun x => Membership.mem f.domain x
    z₂ : Subtype fun x => Membership.mem (f.sup g h).domain x
    hz : Eq ↑z₁ ↑z₂
    ⊢ Eq (HAdd.hAdd (↑f z₁) (↑g 0)) (↑(f.sup g h) z₂)
  -/
  refine (sup_apply h _ _ _ ?_).symm
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    z₁ : Subtype fun x => Membership.mem f.domain x
    z₂ : Subtype fun x => Membership.mem (f.sup g h).domain x
    hz : Eq ↑z₁ ↑z₂
    ⊢ Eq (HAdd.hAdd ↑z₁ ↑0) ↑z₂
  -/
  simpa
  /-
    🎉 no goals
  -/


protected theorem right_le_sup (f g : E →ₗ.[R] F)
    (h : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y) : g ≤ f.sup g h := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    ⊢ LE.le g (f.sup g h)
  -/
  refine ⟨le_sup_right, fun z₁ z₂ hz => ?_⟩
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    z₁ : Subtype fun x => Membership.mem g.domain x
    z₂ : Subtype fun x => Membership.mem (f.sup g h).domain x
    hz : Eq ↑z₁ ↑z₂
    ⊢ Eq (↑g z₁) (↑(f.sup g h) z₂)
  -/
  rw [← zero_add (g _), ← f.map_zero]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    z₁ : Subtype fun x => Membership.mem g.domain x
    z₂ : Subtype fun x => Membership.mem (f.sup g h).domain x
    hz : Eq ↑z₁ ↑z₂
    ⊢ Eq (HAdd.hAdd (↑f 0) (↑g z₁)) (↑(f.sup g h) z₂)
  -/
  refine (sup_apply h _ _ _ ?_).symm
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Subtype fun x => M …
    z₁ : Subtype fun x => Membership.mem g.domain x
    z₂ : Subtype fun x => Membership.mem (f.sup g h).domain x
    hz : Eq ↑z₁ ↑z₂
    ⊢ Eq (HAdd.hAdd ↑0 ↑z₁) ↑z₂
  -/
  simpa
  /-
    🎉 no goals
  -/


protected theorem sup_le {f g h : E →ₗ.[R] F}
    (H : ∀ (x : f.domain) (y : g.domain), (x : E) = y → f x = g y) (fh : f ≤ h) (gh : g ≤ h) :
    f.sup g H ≤ h :=
  have Hf : f ≤ f.sup g H ⊓ h := le_inf (f.left_le_sup g H) fh
  have Hg : g ≤ f.sup g H ⊓ h := le_inf (f.right_le_sup g H) gh
  le_of_eqLocus_ge <| sup_le Hf.1 Hg.1


/-- Hypothesis for `LinearPMap.sup` holds, if `f.domain` is disjoint with `g.domain`. -/
theorem sup_h_of_disjoint (f g : E →ₗ.[R] F) (h : Disjoint f.domain g.domain) (x : f.domain)
    (y : g.domain) (hxy : (x : E) = y) : f x = g y := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : Disjoint f.domain g.domain
    x : Subtype fun x => Membership.mem f.domain x
    y : Subtype fun x => Membership.mem g.domain x
    hxy : Eq ↑x ↑y
    ⊢ Eq (↑f x) (↑g y)
  -/
  rw [disjoint_def] at h
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : E), Membership.mem f.domain x → Membership.mem g.domain x → Eq x 0
    x : Subtype fun x => Membership.mem f.domain x
    y : Subtype fun x => Membership.mem g.domain x
    hxy : Eq ↑x ↑y
    ⊢ Eq (↑f x) (↑g y)
  -/
  have hy : y = 0 := Subtype.eq (h y (hxy ▸ x.2) y.2)
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : E), Membership.mem f.domain x → Membership.mem g.domain x → Eq x 0
    x : Subtype fun x => Membership.mem f.domain x
    y : Subtype fun x => Membership.mem g.domain x
    hxy : Eq ↑x ↑y
    hy : Eq y 0
    ⊢ Eq (↑f x) (↑g y)
  -/
  have hx : x = 0 := Subtype.eq (hxy.trans <| congr_arg _ hy)
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : ∀ (x : E), Membership.mem f.domain x → Membership.mem g.domain x → Eq x 0
    x : Subtype fun x => Membership.mem f.domain x
    y : Subtype fun x => Membership.mem g.domain x
    hxy : Eq ↑x ↑y
    hy : Eq y 0
    hx : Eq x 0
    ⊢ Eq (↑f x) (↑g y)
  -/
  simp [*]
  /-
    🎉 no goals
  -/


instance instZero : Zero (E →ₗ.[R] F) := ⟨⊤, 0⟩


@[simp]
theorem zero_domain : (0 : E →ₗ.[R] F).domain = ⊤ := rfl


@[simp]
theorem zero_apply (x : (⊤ : Submodule R E)) : (0 : E →ₗ.[R] F) x = 0 := rfl


instance instSMul : SMul M (E →ₗ.[R] F) :=
  ⟨fun a f =>
    { domain := f.domain
      toFun := a • f.toFun }⟩


@[simp]
theorem smul_domain (a : M) (f : E →ₗ.[R] F) : (a • f).domain = f.domain :=
  rfl


theorem smul_apply (a : M) (f : E →ₗ.[R] F) (x : (a • f).domain) : (a • f) x = a • f x :=
  rfl


@[simp]
theorem coe_smul (a : M) (f : E →ₗ.[R] F) : ⇑(a • f) = a • ⇑f :=
  rfl


instance instSMulCommClass [SMulCommClass M N F] : SMulCommClass M N (E →ₗ.[R] F) :=
  ⟨fun a b f => ext' <| smul_comm a b f.toFun⟩


instance instIsScalarTower [SMul M N] [IsScalarTower M N F] : IsScalarTower M N (E →ₗ.[R] F) :=
  ⟨fun a b f => ext' <| smul_assoc a b f.toFun⟩


instance instMulAction : MulAction M (E →ₗ.[R] F) where
  smul := (· • ·)
  one_smul := fun ⟨_s, f⟩ => ext' <| one_smul M f
  mul_smul a b f := ext' <| mul_smul a b f.toFun


instance instNeg : Neg (E →ₗ.[R] F) :=
  ⟨fun f => ⟨f.domain, -f.toFun⟩⟩


@[simp]
theorem neg_domain (f : E →ₗ.[R] F) : (-f).domain = f.domain := rfl


@[simp]
theorem neg_apply (f : E →ₗ.[R] F) (x) : (-f) x = -f x :=
  rfl


instance instInvolutiveNeg : InvolutiveNeg (E →ₗ.[R] F) :=
  ⟨fun f => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f : LinearPMap R E F
      ⊢ Eq (Neg.neg (Neg.neg f)) f
    -/
    ext x y hxy
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (Neg.neg (Neg.neg f)).domain x) (Membership.mem f.domain …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h'
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem (Neg.neg (Neg.neg f)).domain x
        y : Subtype fun x => Membership.mem f.domain x
        hxy : Eq ↑x ↑y
        ⊢ Eq (↑(Neg.neg (Neg.neg f)) x) (↑f y)
      -/
    · simp only [neg_apply, neg_neg]
      /-
        case h'
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem (Neg.neg (Neg.neg f)).domain x
        y : Subtype fun x => Membership.mem f.domain x
        hxy : Eq ↑x ↑y
        ⊢ Eq (↑f x) (↑f y)
      -/
      cases x
      /-
        case h'.mk
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        y : Subtype fun x => Membership.mem f.domain x
        val✝ : E
        property✝ : Membership.mem (Neg.neg (Neg.neg f)).domain val✝
        hxy : Eq ↑⟨val✝, property✝⟩ ↑y
        ⊢ Eq (↑f ⟨val✝, property✝⟩) (↑f y)
      -/
      congr⟩
      /-
        🎉 no goals
      -/


instance instAdd : Add (E →ₗ.[R] F) :=
  ⟨fun f g =>
    { domain := f.domain ⊓ g.domain
      toFun := f.toFun.comp (inclusion (inf_le_left : f.domain ⊓ g.domain ≤ _))
        + g.toFun.comp (inclusion (inf_le_right : f.domain ⊓ g.domain ≤ _)) }⟩


theorem add_domain (f g : E →ₗ.[R] F) : (f + g).domain = f.domain ⊓ g.domain := rfl


theorem add_apply (f g : E →ₗ.[R] F) (x : (f.domain ⊓ g.domain : Submodule R E)) :
    (f + g) x = f ⟨x, x.prop.1⟩ + g ⟨x, x.prop.2⟩ := rfl


instance instAddSemigroup : AddSemigroup (E →ₗ.[R] F) :=
  ⟨fun f g h => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g h : LinearPMap R E F
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd f g) h) (HAdd.hAdd f (HAdd.hAdd g h))
    -/
    ext x y hxy
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g h : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (HAdd.hAdd (HAdd.hAdd f g) h).domain x) (Membership.mem  …
      -/
    · simp only [add_domain, inf_assoc]
      /-
        🎉 no goals
      -/
      /-
        case h'
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g h : LinearPMap R E F
        x : Subtype fun x => Membership.mem (HAdd.hAdd (HAdd.hAdd f g) h).domain x
        y : Subtype fun x => Membership.mem (HAdd.hAdd f (HAdd.hAdd g h)).domain x
        hxy : Eq ↑x ↑y
        ⊢ Eq (↑(HAdd.hAdd (HAdd.hAdd f g) h) x) (↑(HAdd.hAdd f (HAdd.hAdd g h)) y)
      -/
    · simp only [add_apply, hxy, add_assoc]⟩
      /-
        🎉 no goals
      -/


instance instAddZeroClass : AddZeroClass (E →ₗ.[R] F) :=
  ⟨fun f => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f : LinearPMap R E F
      ⊢ Eq (HAdd.hAdd 0 f) f
    -/
    ext x y hxy
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (HAdd.hAdd 0 f).domain x) (Membership.mem f.domain x)
      -/
    · simp [add_domain]
      /-
        🎉 no goals
      -/
      /-
        case h'
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem (HAdd.hAdd 0 f).domain x
        y : Subtype fun x => Membership.mem f.domain x
        hxy : Eq ↑x ↑y
        ⊢ Eq (↑(HAdd.hAdd 0 f) x) (↑f y)
      -/
    · simp only [add_apply, hxy, zero_apply, zero_add],
      /-
        🎉 no goals
      -/
  fun f => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f : LinearPMap R E F
      ⊢ Eq (HAdd.hAdd f 0) f
    -/
    ext x y hxy
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (HAdd.hAdd f 0).domain x) (Membership.mem f.domain x)
      -/
    · simp [add_domain]
      /-
        🎉 no goals
      -/
      /-
        case h'
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f : LinearPMap R E F
        x : Subtype fun x => Membership.mem (HAdd.hAdd f 0).domain x
        y : Subtype fun x => Membership.mem f.domain x
        hxy : Eq ↑x ↑y
        ⊢ Eq (↑(HAdd.hAdd f 0) x) (↑f y)
      -/
    · simp only [add_apply, hxy, zero_apply, add_zero]⟩
      /-
        🎉 no goals
      -/


instance instAddMonoid : AddMonoid (E →ₗ.[R] F) where
  zero_add f := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f : LinearPMap R E F
      ⊢ Eq (HAdd.hAdd 0 f) f
    -/
    simp
    /-
      🎉 no goals
    -/
  add_zero := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      ⊢ ∀ (a : LinearPMap R E F), Eq (HAdd.hAdd a 0) a
    -/
    simp
    /-
      🎉 no goals
    -/
  nsmul := nsmulRec


instance instAddCommMonoid : AddCommMonoid (E →ₗ.[R] F) :=
  ⟨fun f g => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      ⊢ Eq (HAdd.hAdd f g) (HAdd.hAdd g f)
    -/
    ext x y hxy
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (HAdd.hAdd f g).domain x) (Membership.mem (HAdd.hAdd g f …
      -/
    · simp only [add_domain, inf_comm]
      /-
        🎉 no goals
      -/
      /-
        case h'
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        x : Subtype fun x => Membership.mem (HAdd.hAdd f g).domain x
        y : Subtype fun x => Membership.mem (HAdd.hAdd g f).domain x
        hxy : Eq ↑x ↑y
        ⊢ Eq (↑(HAdd.hAdd f g) x) (↑(HAdd.hAdd g f) y)
      -/
    · simp only [add_apply, hxy, add_comm]⟩
      /-
        🎉 no goals
      -/


instance instVAdd : VAdd (E →ₗ[R] F) (E →ₗ.[R] F) :=
  ⟨fun f g =>
    { domain := g.domain
      toFun := f.comp g.domain.subtype + g.toFun }⟩


@[simp]
theorem vadd_domain (f : E →ₗ[R] F) (g : E →ₗ.[R] F) : (f +ᵥ g).domain = g.domain :=
  rfl


theorem vadd_apply (f : E →ₗ[R] F) (g : E →ₗ.[R] F) (x : (f +ᵥ g).domain) :
    (f +ᵥ g) x = f x + g x :=
  rfl


@[simp]
theorem coe_vadd (f : E →ₗ[R] F) (g : E →ₗ.[R] F) : ⇑(f +ᵥ g) = ⇑(f.comp g.domain.subtype) + ⇑g :=
  rfl


instance instAddAction : AddAction (E →ₗ[R] F) (E →ₗ.[R] F) where
  vadd := (· +ᵥ ·)
  zero_vadd := fun ⟨_s, _f⟩ => ext' <| zero_add _
  add_vadd := fun _f₁ _f₂ ⟨_s, _g⟩ => ext' <| LinearMap.ext fun _x => add_assoc _ _ _


instance instSub : Sub (E →ₗ.[R] F) :=
  ⟨fun f g =>
    { domain := f.domain ⊓ g.domain
      toFun := f.toFun.comp (inclusion (inf_le_left : f.domain ⊓ g.domain ≤ _))
        - g.toFun.comp (inclusion (inf_le_right : f.domain ⊓ g.domain ≤ _)) }⟩


theorem sub_domain (f g : E →ₗ.[R] F) : (f - g).domain = f.domain ⊓ g.domain := rfl


theorem sub_apply (f g : E →ₗ.[R] F) (x : (f.domain ⊓ g.domain : Submodule R E)) :
    (f - g) x = f ⟨x, x.prop.1⟩ - g ⟨x, x.prop.2⟩ := rfl


instance instSubtractionCommMonoid : SubtractionCommMonoid (E →ₗ.[R] F) where
  add_comm := add_comm
  sub_eq_add_neg f g := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      ⊢ Eq (HSub.hSub f g) (HAdd.hAdd f (Neg.neg g))
    -/
    ext x y h
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (HSub.hSub f g).domain x) (Membership.mem (HAdd.hAdd f ( …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case h'
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      x : Subtype fun x => Membership.mem (HSub.hSub f g).domain x
      y : Subtype fun x => Membership.mem (HAdd.hAdd f (Neg.neg g)).domain x
      h : Eq ↑x ↑y
      ⊢ Eq (↑(HSub.hSub f g) x) (↑(HAdd.hAdd f (Neg.neg g)) y)
    -/
    simp [sub_apply, add_apply, neg_apply, ← sub_eq_add_neg, h]
    /-
      🎉 no goals
    -/
  neg_neg := neg_neg
  neg_add_rev f g := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      ⊢ Eq (Neg.neg (HAdd.hAdd f g)) (HAdd.hAdd (Neg.neg g) (Neg.neg f))
    -/
    ext x y h
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        x : E
        ⊢ Iff (Membership.mem (Neg.neg (HAdd.hAdd f g)).domain x) (Membership.mem (HAd …
      -/
    · simp [add_domain, sub_domain, neg_domain, And.comm]
      /-
        🎉 no goals
      -/
    /-
      case h'
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      x : Subtype fun x => Membership.mem (Neg.neg (HAdd.hAdd f g)).domain x
      y : Subtype fun x => Membership.mem (HAdd.hAdd (Neg.neg g) (Neg.neg f)).domain x
      h : Eq ↑x ↑y
      ⊢ Eq (↑(Neg.neg (HAdd.hAdd f g)) x) (↑(HAdd.hAdd (Neg.neg g) (Neg.neg f)) y)
    -/
    simp [sub_apply, add_apply, neg_apply, ← sub_eq_add_neg, h]
    /-
      🎉 no goals
    -/
  neg_eq_of_add f g h' := by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      h' : Eq (HAdd.hAdd f g) 0
      ⊢ Eq (Neg.neg f) g
    -/
    ext x y h
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        h' : Eq (HAdd.hAdd f g) 0
        x : E
        ⊢ Iff (Membership.mem (Neg.neg f).domain x) (Membership.mem g.domain x)
      -/
    · have : (0 : E →ₗ.[R] F).domain = ⊤ := zero_domain
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        h' : Eq (HAdd.hAdd f g) 0
        x : E
        this : Eq (LinearPMap.domain 0) Top.top
        ⊢ Iff (Membership.mem (Neg.neg f).domain x) (Membership.mem g.domain x)
      -/
      simp only [← h', add_domain, inf_eq_top_iff] at this
      /-
        case h.h
        R : Type u_1
        inst✝⁶ : Ring R
        E : Type u_2
        inst✝⁵ : AddCommGroup E
        inst✝⁴ : Module R E
        F : Type u_3
        inst✝³ : AddCommGroup F
        inst✝² : Module R F
        G : Type u_4
        inst✝¹ : AddCommGroup G
        inst✝ : Module R G
        f g : LinearPMap R E F
        h' : Eq (HAdd.hAdd f g) 0
        x : E
        this : And (Eq f.domain Top.top) (Eq g.domain Top.top)
        ⊢ Iff (Membership.mem (Neg.neg f).domain x) (Membership.mem g.domain x)
      -/
      rw [neg_domain, this.1, this.2]
      /-
        🎉 no goals
      -/
    /-
      case h'
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      h' : Eq (HAdd.hAdd f g) 0
      x : Subtype fun x => Membership.mem (Neg.neg f).domain x
      y : Subtype fun x => Membership.mem g.domain x
      h : Eq ↑x ↑y
      ⊢ Eq (↑(Neg.neg f) x) (↑g y)
    -/
    simp only [inf_coe, neg_domain, Eq.ndrec, Int.ofNat_eq_coe, neg_apply]
    /-
      case h'
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      h' : Eq (HAdd.hAdd f g) 0
      x : Subtype fun x => Membership.mem (Neg.neg f).domain x
      y : Subtype fun x => Membership.mem g.domain x
      h : Eq ↑x ↑y
      ⊢ Eq (Neg.neg (↑f x)) (↑g y)
    -/
    rw [ext_iff] at h'
    /-
      case h'
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      h' : Exists fun _domain_eq => ∀ ⦃x : Subtype fun x => Membership.mem (HAdd.hAd …
      x : Subtype fun x => Membership.mem (Neg.neg f).domain x
      y : Subtype fun x => Membership.mem g.domain x
      h : Eq ↑x ↑y
      ⊢ Eq (Neg.neg (↑f x)) (↑g y)
    -/
    rcases h' with ⟨hdom, h'⟩
    /-
      case h'.intro
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      x : Subtype fun x => Membership.mem (Neg.neg f).domain x
      y : Subtype fun x => Membership.mem g.domain x
      h : Eq ↑x ↑y
      hdom : Eq (HAdd.hAdd f g).domain (LinearPMap.domain 0)
      h' : ∀ ⦃x : Subtype fun x => Membership.mem (HAdd.hAdd f g).domain x⦄ ⦃y : Sub …
      ⊢ Eq (Neg.neg (↑f x)) (↑g y)
    -/
    rw [zero_domain] at hdom
    simp only [inf_coe, neg_domain, Eq.ndrec, Int.ofNat_eq_coe, zero_domain, top_coe, zero_apply,
      Subtype.forall, mem_top, forall_true_left, forall_eq'] at h'
    /-
      case h'.intro
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      x : Subtype fun x => Membership.mem (Neg.neg f).domain x
      y : Subtype fun x => Membership.mem g.domain x
      h : Eq ↑x ↑y
      hdom : Eq (HAdd.hAdd f g).domain Top.top
      h' : ∀ (a : E) (b : Membership.mem (HAdd.hAdd f g).domain a), Eq (↑(HAdd.hAdd  …
      ⊢ Eq (Neg.neg (↑f x)) (↑g y)
    -/
    specialize h' x.1 (by simp [hdom])
    simp only [inf_coe, neg_domain, Eq.ndrec, Int.ofNat_eq_coe, add_apply, Subtype.coe_eta,
      ← neg_eq_iff_add_eq_zero] at h'
    /-
      case h'.intro
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      f g : LinearPMap R E F
      x : Subtype fun x => Membership.mem (Neg.neg f).domain x
      y : Subtype fun x => Membership.mem g.domain x
      h : Eq ↑x ↑y
      hdom : Eq (HAdd.hAdd f g).domain Top.top
      h' : Eq (Neg.neg (↑f x)) (↑g ⟨↑x, ⋯⟩)
      ⊢ Eq (Neg.neg (↑f x)) (↑g y)
    -/
    rw [h', h]
    /-
      🎉 no goals
    -/
  zsmul := zsmulRec


/-- Extend a `LinearPMap` to `f.domain ⊔ K ∙ x`. -/
noncomputable def supSpanSingleton (f : E →ₗ.[K] F) (x : E) (y : F) (hx : x ∉ f.domain) :
    E →ₗ.[K] F :=
  -- Porting note: `simpa [..]` → `simp [..]; exact ..`
  f.sup (mkSpanSingleton x y fun h₀ => hx <| h₀.symm ▸ f.domain.zero_mem) <|
                                /-
                                  R : Type u_1
                                  inst✝⁹ : Ring R
                                  E : Type u_2
                                  inst✝⁸ : AddCommGroup E
                                  inst✝⁷ : Module R E
                                  F : Type u_3
                                  inst✝⁶ : AddCommGroup F
                                  inst✝⁵ : Module R F
                                  G : Type u_4
                                  inst✝⁴ : AddCommGroup G
                                  inst✝³ : Module R G
                                  K : Type u_5
                                  inst✝² : DivisionRing K
                                  inst✝¹ : Module K E
                                  inst✝ : Module K F
                                  f : LinearPMap K E F
                                  x : E
                                  y : F
                                  hx : Not (Membership.mem f.domain x)
                                  ⊢ Disjoint f.domain (LinearPMap.mkSpanSingleton x y ⋯).domain
                                -/
    sup_h_of_disjoint _ _ <| by simpa [disjoint_span_singleton] using fun h ↦ False.elim <| hx h
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem domain_supSpanSingleton (f : E →ₗ.[K] F) (x : E) (y : F) (hx : x ∉ f.domain) :
    (f.supSpanSingleton x y hx).domain = f.domain ⊔ K ∙ x :=
  rfl


@[simp]
theorem supSpanSingleton_apply_mk (f : E →ₗ.[K] F) (x : E) (y : F) (hx : x ∉ f.domain) (x' : E)
    (hx' : x' ∈ f.domain) (c : K) :
    f.supSpanSingleton x y hx
        ⟨x' + c • x, mem_sup.2 ⟨x', hx', _, mem_span_singleton.2 ⟨c, rfl⟩, rfl⟩⟩ =
      f ⟨x', hx'⟩ + c • y := by
  -- Porting note: `erw [..]; rfl; exact ..` → `erw [..]; exact ..; rfl`
  -- That is, the order of the side goals generated by `erw` changed.
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    F : Type u_3
    inst✝³ : AddCommGroup F
    K : Type u_5
    inst✝² : DivisionRing K
    inst✝¹ : Module K E
    inst✝ : Module K F
    f : LinearPMap K E F
    x : E
    y : F
    hx : Not (Membership.mem f.domain x)
    x' : E
    hx' : Membership.mem f.domain x'
    c : K
    ⊢ Eq (↑(f.supSpanSingleton x y hx) ⟨HAdd.hAdd x' (HSMul.hSMul c x), ⋯⟩) (HAdd. …
  -/
  erw [sup_apply _ ⟨x', hx'⟩ ⟨c • x, _⟩, mkSpanSingleton'_apply]
    /-
      case h
      E : Type u_2
      inst✝⁴ : AddCommGroup E
      F : Type u_3
      inst✝³ : AddCommGroup F
      K : Type u_5
      inst✝² : DivisionRing K
      inst✝¹ : Module K E
      inst✝ : Module K F
      f : LinearPMap K E F
      x : E
      y : F
      hx : Not (Membership.mem f.domain x)
      x' : E
      hx' : Membership.mem f.domain x'
      c : K
      ⊢ Membership.mem (LinearPMap.mkSpanSingleton' x y ⋯).domain (HSMul.hSMul c x)
    -/
  · exact mem_span_singleton.2 ⟨c, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case hz
      E : Type u_2
      inst✝⁴ : AddCommGroup E
      F : Type u_3
      inst✝³ : AddCommGroup F
      K : Type u_5
      inst✝² : DivisionRing K
      inst✝¹ : Module K E
      inst✝ : Module K F
      f : LinearPMap K E F
      x : E
      y : F
      hx : Not (Membership.mem f.domain x)
      x' : E
      hx' : Membership.mem f.domain x'
      c : K
      ⊢ Eq (HAdd.hAdd ↑⟨x', hx'⟩ ↑⟨HSMul.hSMul c x, ⋯⟩) ↑⟨HAdd.hAdd x' (HSMul.hSMul  …
    -/
  · rfl
    /-
      🎉 no goals
    -/


private theorem sSup_aux (c : Set (E →ₗ.[R] F)) (hc : DirectedOn (· ≤ ·) c) :
    ∃ f : ↥(sSup (domain '' c)) →ₗ[R] F, (⟨_, f⟩ : E →ₗ.[R] F) ∈ upperBounds c := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    c : Set (LinearPMap R E F)
    hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
    ⊢ Exists fun f => Membership.mem (upperBounds c) { domain := SupSet.sSup (Set. …
  -/
  rcases c.eq_empty_or_nonempty with ceq | cne
    /-
      case inl
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      ceq : Eq c EmptyCollection.emptyCollection
      ⊢ Exists fun f => Membership.mem (upperBounds c) { domain := SupSet.sSup (Set. …
    -/
  · subst c
    /-
      case inl
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) EmptyCollection.emptyCollection
      ⊢ Exists fun f => Membership.mem (upperBounds EmptyCollection.emptyCollection) …
    -/
    simp
    /-
      🎉 no goals
    -/
  have hdir : DirectedOn (· ≤ ·) (domain '' c) :=
    directedOn_image.2 (hc.mono @(domain_mono.monotone))
  have P : ∀ x : ↥(sSup (domain '' c)), { p : c // (x : E) ∈ p.val.domain } := by
    rintro x
    apply Classical.indefiniteDescription
    have := (mem_sSup_of_directed (cne.image _) hdir).1 x.2
    -- Porting note: + `← bex_def`
    rwa [Set.exists_mem_image, ← bex_def, SetCoe.exists'] at this
  /-
    case inr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    c : Set (LinearPMap R E F)
    hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
    cne : c.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
    P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
    ⊢ Exists fun f => Membership.mem (upperBounds c) { domain := SupSet.sSup (Set. …
  -/
  set f : ↥(sSup (domain '' c)) → F := fun x => (P x).val.val ⟨x, (P x).property⟩
  have f_eq : ∀ (p : c) (x : ↥(sSup (domain '' c))) (y : p.1.1) (_hxy : (x : E) = y),
      f x = p.1 y := by
    intro p x y hxy
    rcases hc (P x).1.1 (P x).1.2 p.1 p.2 with ⟨q, _hqc, hxq, hpq⟩
    -- Porting note: `refine' ..; exacts [inclusion hpq.1 y, hxy, rfl]`
    --               → `refine' .. <;> [skip; exact inclusion hpq.1 y; rfl]; exact hxy`
    convert (hxq.2 _).trans (hpq.2 _).symm <;> [skip; exact inclusion hpq.1 y; rfl]; exact hxy
  /-
    case inr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    c : Set (LinearPMap R E F)
    hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
    cne : c.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
    P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
    f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
    f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
    ⊢ Exists fun f => Membership.mem (upperBounds c) { domain := SupSet.sSup (Set. …
  -/
  use { toFun := f, map_add' := ?_, map_smul' := ?_ }, ?_
    /-
      case w.refine_1
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      ⊢ ∀ (x y : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap. …
    -/
  · intro x y
    /-
      case w.refine_1
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      x y : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domai …
      ⊢ Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    -/
    rcases hc (P x).1.1 (P x).1.2 (P y).1.1 (P y).1.2 with ⟨p, hpc, hpx, hpy⟩
    /-
      case w.refine_1.intro.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      x y : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domai …
      p : LinearPMap R E F
      hpc : Membership.mem c p
      hpx : LE.le (↑↑(P x)) p
      hpy : LE.le (↑↑(P y)) p
      ⊢ Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    -/
    set x' := inclusion hpx.1 ⟨x, (P x).2⟩
    /-
      case w.refine_1.intro.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      x y : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domai …
      p : LinearPMap R E F
      hpc : Membership.mem c p
      hpx : LE.le (↑↑(P x)) p
      hpy : LE.le (↑↑(P y)) p
      x' : Subtype fun x => Membership.mem p.domain x := (Submodule.inclusion ⋯) ⟨↑x …
      ⊢ Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    -/
    set y' := inclusion hpy.1 ⟨y, (P y).2⟩
    rw [f_eq ⟨p, hpc⟩ x x' rfl, f_eq ⟨p, hpc⟩ y y' rfl, f_eq ⟨p, hpc⟩ (x + y) (x' + y') rfl,
      map_add]
    /-
      case w.refine_2
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      ⊢ ∀ (m : R) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image Linea …
    -/
  · intro c x
    /-
      case w.refine_2
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c✝ : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c✝
      cne : c✝.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c✝)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c✝) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.imag …
      c : R
      x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain  …
      ⊢ Eq ({ toFun := f, map_add' := ⋯ }.toFun (HSMul.hSMul c x)) (HSMul.hSMul ((Ri …
    -/
    simp only [RingHom.id_apply]
    /-
      case w.refine_2
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c✝ : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c✝
      cne : c✝.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c✝)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c✝) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.imag …
      c : R
      x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain  …
      ⊢ Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
    -/
    rw [f_eq (P x).1 (c • x) (c • ⟨x, (P x).2⟩) rfl, ← map_smul]
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      ⊢ Membership.mem (upperBounds c) { domain := SupSet.sSup (Set.image LinearPMap …
    -/
  · intro p hpc
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      p : LinearPMap R E F
      hpc : Membership.mem c p
      ⊢ LE.le p { domain := SupSet.sSup (Set.image LinearPMap.domain c), toFun := {  …
    -/
    refine ⟨le_sSup <| Set.mem_image_of_mem domain hpc, fun x y hxy => Eq.symm ?_⟩
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      c : Set (LinearPMap R E F)
      hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
      cne : c.Nonempty
      hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) (Set.image LinearPMap.domain c)
      P : (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.do …
      f : (Subtype fun x => Membership.mem (SupSet.sSup (Set.image LinearPMap.domain …
      f_eq : ∀ (p : ↑c) (x : Subtype fun x => Membership.mem (SupSet.sSup (Set.image …
      p : LinearPMap R E F
      hpc : Membership.mem c p
      x : Subtype fun x => Membership.mem p.domain x
      y : Subtype fun x => Membership.mem { domain := SupSet.sSup (Set.image LinearP …
      hxy : Eq ↑x ↑y
      ⊢ Eq (↑{ domain := SupSet.sSup (Set.image LinearPMap.domain c), toFun := { toF …
    -/
    exact f_eq ⟨p, hpc⟩ _ _ hxy.symm
    /-
      🎉 no goals
    -/


protected noncomputable def sSup (c : Set (E →ₗ.[R] F)) (hc : DirectedOn (· ≤ ·) c) : E →ₗ.[R] F :=
  ⟨_, Classical.choose <| sSup_aux c hc⟩


protected theorem le_sSup {c : Set (E →ₗ.[R] F)} (hc : DirectedOn (· ≤ ·) c) {f : E →ₗ.[R] F}
    (hf : f ∈ c) : f ≤ LinearPMap.sSup c hc :=
  Classical.choose_spec (sSup_aux c hc) hf


protected theorem sSup_le {c : Set (E →ₗ.[R] F)} (hc : DirectedOn (· ≤ ·) c) {g : E →ₗ.[R] F}
    (hg : ∀ f ∈ c, f ≤ g) : LinearPMap.sSup c hc ≤ g :=
  le_of_eqLocus_ge <|
    sSup_le fun _ ⟨f, hf, Eq⟩ =>
      Eq ▸
        have : f ≤ LinearPMap.sSup c hc ⊓ g := le_inf (LinearPMap.le_sSup _ hf) (hg f hf)
        this.1


protected theorem sSup_apply {c : Set (E →ₗ.[R] F)} (hc : DirectedOn (· ≤ ·) c) {l : E →ₗ.[R] F}
    (hl : l ∈ c) (x : l.domain) :
    (LinearPMap.sSup c hc) ⟨x, (LinearPMap.le_sSup hc hl).1 x.2⟩ = l x := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    c : Set (LinearPMap R E F)
    hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
    l : LinearPMap R E F
    hl : Membership.mem c l
    x : Subtype fun x => Membership.mem l.domain x
    ⊢ Eq (↑(LinearPMap.sSup c hc) ⟨↑x, ⋯⟩) (↑l x)
  -/
  symm
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    c : Set (LinearPMap R E F)
    hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
    l : LinearPMap R E F
    hl : Membership.mem c l
    x : Subtype fun x => Membership.mem l.domain x
    ⊢ Eq (↑l x) (↑(LinearPMap.sSup c hc) ⟨↑x, ⋯⟩)
  -/
  apply (Classical.choose_spec (sSup_aux c hc) hl).2
  /-
    case _h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    c : Set (LinearPMap R E F)
    hc : DirectedOn (fun x1 x2 => LE.le x1 x2) c
    l : LinearPMap R E F
    hl : Membership.mem c l
    x : Subtype fun x => Membership.mem l.domain x
    ⊢ Eq ↑x ↑⟨↑x, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Restrict a linear map to a submodule, reinterpreting the result as a `LinearPMap`. -/
def toPMap (f : E →ₗ[R] F) (p : Submodule R E) : E →ₗ.[R] F :=
  ⟨p, f.comp p.subtype⟩


@[simp]
theorem toPMap_apply (f : E →ₗ[R] F) (p : Submodule R E) (x : p) : f.toPMap p x = f x :=
  rfl


@[simp]
theorem toPMap_domain (f : E →ₗ[R] F) (p : Submodule R E) : (f.toPMap p).domain = p :=
  rfl


/-- Compose a linear map with a `LinearPMap` -/
def compPMap (g : F →ₗ[R] G) (f : E →ₗ.[R] F) : E →ₗ.[R] G where
  domain := f.domain
  toFun := g.comp f.toFun


@[simp]
theorem compPMap_apply (g : F →ₗ[R] G) (f : E →ₗ.[R] F) (x) : g.compPMap f x = g (f x) :=
  rfl


/-- Restrict codomain of a `LinearPMap` -/
def codRestrict (f : E →ₗ.[R] F) (p : Submodule R F) (H : ∀ x, f x ∈ p) : E →ₗ.[R] p where
  domain := f.domain
  toFun := f.toFun.codRestrict p H


/-- Compose two `LinearPMap`s -/
def comp (g : F →ₗ.[R] G) (f : E →ₗ.[R] F) (H : ∀ x : f.domain, f x ∈ g.domain) : E →ₗ.[R] G :=
  g.toFun.compPMap <| f.codRestrict _ H


/-- `f.coprod g` is the partially defined linear map defined on `f.domain × g.domain`,
and sending `p` to `f p.1 + g p.2`. -/
def coprod (f : E →ₗ.[R] G) (g : F →ₗ.[R] G) : E × F →ₗ.[R] G where
  domain := f.domain.prod g.domain
  toFun :=
    -- Porting note: This is just
    -- `(f.comp (LinearPMap.fst f.domain g.domain) fun x => x.2.1).toFun +`
    -- `  (g.comp (LinearPMap.snd f.domain g.domain) fun x => x.2.2).toFun`,
    HAdd.hAdd
      (α := f.domain.prod g.domain →ₗ[R] G)
      (β := f.domain.prod g.domain →ₗ[R] G)
      (f.comp (LinearPMap.fst f.domain g.domain) fun x => x.2.1).toFun
      (g.comp (LinearPMap.snd f.domain g.domain) fun x => x.2.2).toFun


@[simp]
theorem coprod_apply (f : E →ₗ.[R] G) (g : F →ₗ.[R] G) (x) :
    f.coprod g x = f ⟨(x : E × F).1, x.2.1⟩ + g ⟨(x : E × F).2, x.2.2⟩ :=
  rfl


/-- Restrict a partially defined linear map to a submodule of `E` contained in `f.domain`. -/
def domRestrict (f : E →ₗ.[R] F) (S : Submodule R E) : E →ₗ.[R] F :=
                                                       /-
                                                         R : Type u_1
                                                         inst✝⁶ : Ring R
                                                         E : Type u_2
                                                         inst✝⁵ : AddCommGroup E
                                                         inst✝⁴ : Module R E
                                                         F : Type u_3
                                                         inst✝³ : AddCommGroup F
                                                         inst✝² : Module R F
                                                         G : Type u_4
                                                         inst✝¹ : AddCommGroup G
                                                         inst✝ : Module R G
                                                         f : LinearPMap R E F
                                                         S : Submodule R E
                                                         ⊢ LE.le (Min.min S f.domain) f.domain
                                                       -/
  ⟨S ⊓ f.domain, f.toFun.comp (Submodule.inclusion (by simp))⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem domRestrict_domain (f : E →ₗ.[R] F) {S : Submodule R E} :
    (f.domRestrict S).domain = S ⊓ f.domain :=
  rfl


theorem domRestrict_apply {f : E →ₗ.[R] F} {S : Submodule R E} ⦃x : ↥(S ⊓ f.domain)⦄ ⦃y : f.domain⦄
    (h : (x : E) = y) : f.domRestrict S x = f y := by
  have : Submodule.inclusion (by simp) x = y := by
    ext
    simp [h]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    S : Submodule R E
    x : Subtype fun x => Membership.mem (Min.min S f.domain) x
    y : Subtype fun x => Membership.mem f.domain x
    h : Eq ↑x ↑y
    this : Eq ((Submodule.inclusion ⋯) x) y
    ⊢ Eq (↑(f.domRestrict S) x) (↑f y)
  -/
  rw [← this]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    S : Submodule R E
    x : Subtype fun x => Membership.mem (Min.min S f.domain) x
    y : Subtype fun x => Membership.mem f.domain x
    h : Eq ↑x ↑y
    this : Eq ((Submodule.inclusion ⋯) x) y
    ⊢ Eq (↑(f.domRestrict S) x) (↑f ((Submodule.inclusion ⋯) x))
  -/
  exact LinearPMap.mk_apply _ _ _
  /-
    🎉 no goals
  -/


theorem domRestrict_le {f : E →ₗ.[R] F} {S : Submodule R E} : f.domRestrict S ≤ f :=
      /-
        R : Type u_1
        inst✝⁴ : Ring R
        E : Type u_2
        inst✝³ : AddCommGroup E
        inst✝² : Module R E
        F : Type u_3
        inst✝¹ : AddCommGroup F
        inst✝ : Module R F
        f : LinearPMap R E F
        S : Submodule R E
        ⊢ LE.le (f.domRestrict S).domain f.domain
      -/
  ⟨by simp, fun _ _ hxy => domRestrict_apply hxy⟩
      /-
        🎉 no goals
      -/


/-- The graph of a `LinearPMap` viewed as a submodule on `E × F`. -/
def graph (f : E →ₗ.[R] F) : Submodule R (E × F) :=
  f.toFun.graph.map (f.domain.subtype.prodMap (LinearMap.id : F →ₗ[R] F))


theorem mem_graph_iff' (f : E →ₗ.[R] F) {x : E × F} :
                                                      /-
                                                        R : Type u_1
                                                        inst✝⁴ : Ring R
                                                        E : Type u_2
                                                        inst✝³ : AddCommGroup E
                                                        inst✝² : Module R E
                                                        F : Type u_3
                                                        inst✝¹ : AddCommGroup F
                                                        inst✝ : Module R F
                                                        f : LinearPMap R E F
                                                        x : Prod E F
                                                        ⊢ Iff (Membership.mem f.graph x) (Exists fun y => Eq { fst := ↑y, snd := ↑f y  …
                                                      -/
    x ∈ f.graph ↔ ∃ y : f.domain, (↑y, f y) = x := by simp [graph]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem mem_graph_iff (f : E →ₗ.[R] F) {x : E × F} :
    x ∈ f.graph ↔ ∃ y : f.domain, (↑y : E) = x.1 ∧ f y = x.2 := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : Prod E F
    ⊢ Iff (Membership.mem f.graph x) (Exists fun y => And (Eq (↑y) x.1) (Eq (↑f y) …
  -/
  cases x
  /-
    case mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    fst✝ : E
    snd✝ : F
    ⊢ Iff (Membership.mem f.graph { fst := fst✝, snd := snd✝ }) (Exists fun y => A …
  -/
  simp_rw [mem_graph_iff', Prod.mk.inj_iff]
  /-
    🎉 no goals
  -/


/-- The tuple `(x, f x)` is contained in the graph of `f`. -/
                                                                                   /-
                                                                                     R : Type u_1
                                                                                     inst✝⁴ : Ring R
                                                                                     E : Type u_2
                                                                                     inst✝³ : AddCommGroup E
                                                                                     inst✝² : Module R E
                                                                                     F : Type u_3
                                                                                     inst✝¹ : AddCommGroup F
                                                                                     inst✝ : Module R F
                                                                                     f : LinearPMap R E F
                                                                                     x : Subtype fun x => Membership.mem f.domain x
                                                                                     ⊢ Membership.mem f.graph { fst := ↑x, snd := ↑f x }
                                                                                   -/
theorem mem_graph (f : E →ₗ.[R] F) (x : domain f) : ((x : E), f x) ∈ f.graph := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem graph_map_fst_eq_domain (f : E →ₗ.[R] F) :
    f.graph.map (LinearMap.fst R E F) = f.domain := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    ⊢ Eq (Submodule.map (LinearMap.fst R E F) f.graph) f.domain
  -/
  ext x
  simp only [Submodule.mem_map, mem_graph_iff, Subtype.exists, exists_and_left, exists_eq_left,
    LinearMap.fst_apply, Prod.exists, exists_and_right, exists_eq_right]
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    ⊢ Iff (Exists fun x_1 => Exists fun h => Eq (↑f ⟨x, ⋯⟩) x_1) (Membership.mem f …
  -/
  constructor <;> intro h
    /-
      case h.mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      h : Exists fun x_1 => Exists fun h => Eq (↑f ⟨x, ⋯⟩) x_1
      ⊢ Membership.mem f.domain x
    -/
  · rcases h with ⟨x, hx, _⟩
    /-
      case h.mp.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x✝ : E
      x : F
      hx : Membership.mem f.domain x✝
      h✝ : Eq (↑f ⟨x✝, ⋯⟩) x
      ⊢ Membership.mem f.domain x✝
    -/
    exact hx
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      h : Membership.mem f.domain x
      ⊢ Exists fun x_1 => Exists fun h => Eq (↑f ⟨x, ⋯⟩) x_1
    -/
  · use f ⟨x, h⟩
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      h : Membership.mem f.domain x
      ⊢ Exists fun h_1 => Eq (↑f ⟨x, ⋯⟩) (↑f ⟨x, h⟩)
    -/
    simp only [h, exists_const]
    /-
      🎉 no goals
    -/


theorem graph_map_snd_eq_range (f : E →ₗ.[R] F) :
                                                                      /-
                                                                        R : Type u_1
                                                                        inst✝⁴ : Ring R
                                                                        E : Type u_2
                                                                        inst✝³ : AddCommGroup E
                                                                        inst✝² : Module R E
                                                                        F : Type u_3
                                                                        inst✝¹ : AddCommGroup F
                                                                        inst✝ : Module R F
                                                                        f : LinearPMap R E F
                                                                        ⊢ Eq (Submodule.map (LinearMap.snd R E F) f.graph) (LinearMap.range f.toFun)
                                                                      -/
    f.graph.map (LinearMap.snd R E F) = LinearMap.range f.toFun := by ext; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The graph of `z • f` as a pushforward. -/
theorem smul_graph (f : E →ₗ.[R] F) (z : M) :
    (z • f).graph =
      f.graph.map ((LinearMap.id : E →ₗ[R] E).prodMap (z • (LinearMap.id : F →ₗ[R] F))) := by
  /-
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    ⊢ Eq (HSMul.hSMul z f).graph (Submodule.map (LinearMap.id.prodMap (HSMul.hSMul …
  -/
  ext x; cases' x with x_fst x_snd
  /-
    case h.mk
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    ⊢ Iff (Membership.mem (HSMul.hSMul z f).graph { fst := x_fst, snd := x_snd })  …
  -/
  constructor <;> intro h
    /-
      case h.mk.mp
      R : Type u_1
      inst✝⁷ : Ring R
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module R E
      F : Type u_3
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module R F
      M : Type u_5
      inst✝² : Monoid M
      inst✝¹ : DistribMulAction M F
      inst✝ : SMulCommClass R M F
      f : LinearPMap R E F
      z : M
      x_fst : E
      x_snd : F
      h : Membership.mem (HSMul.hSMul z f).graph { fst := x_fst, snd := x_snd }
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (HSMul.hSMul z LinearMap …
    -/
  · rw [mem_graph_iff] at h
    /-
      case h.mk.mp
      R : Type u_1
      inst✝⁷ : Ring R
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module R E
      F : Type u_3
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module R F
      M : Type u_5
      inst✝² : Monoid M
      inst✝¹ : DistribMulAction M F
      inst✝ : SMulCommClass R M F
      f : LinearPMap R E F
      z : M
      x_fst : E
      x_snd : F
      h : Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(HSMul. …
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (HSMul.hSMul z LinearMap …
    -/
    rcases h with ⟨y, hy, h⟩
    /-
      case h.mk.mp.intro.intro
      R : Type u_1
      inst✝⁷ : Ring R
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module R E
      F : Type u_3
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module R F
      M : Type u_5
      inst✝² : Monoid M
      inst✝¹ : DistribMulAction M F
      inst✝ : SMulCommClass R M F
      f : LinearPMap R E F
      z : M
      x_fst : E
      x_snd : F
      y : Subtype fun x => Membership.mem (HSMul.hSMul z f).domain x
      hy : Eq ↑y { fst := x_fst, snd := x_snd }.1
      h : Eq (↑(HSMul.hSMul z f) y) { fst := x_fst, snd := x_snd }.2
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (HSMul.hSMul z LinearMap …
    -/
    rw [LinearPMap.smul_apply] at h
    /-
      case h.mk.mp.intro.intro
      R : Type u_1
      inst✝⁷ : Ring R
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module R E
      F : Type u_3
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module R F
      M : Type u_5
      inst✝² : Monoid M
      inst✝¹ : DistribMulAction M F
      inst✝ : SMulCommClass R M F
      f : LinearPMap R E F
      z : M
      x_fst : E
      x_snd : F
      y : Subtype fun x => Membership.mem (HSMul.hSMul z f).domain x
      hy : Eq ↑y { fst := x_fst, snd := x_snd }.1
      h : Eq (HSMul.hSMul z (↑f y)) { fst := x_fst, snd := x_snd }.2
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (HSMul.hSMul z LinearMap …
    -/
    rw [Submodule.mem_map]
    simp only [mem_graph_iff, LinearMap.prodMap_apply, LinearMap.id_coe, id,
      LinearMap.smul_apply, Prod.mk.inj_iff, Prod.exists, exists_exists_and_eq_and]
    /-
      case h.mk.mp.intro.intro
      R : Type u_1
      inst✝⁷ : Ring R
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module R E
      F : Type u_3
      inst✝⁴ : AddCommGroup F
      inst✝³ : Module R F
      M : Type u_5
      inst✝² : Monoid M
      inst✝¹ : DistribMulAction M F
      inst✝ : SMulCommClass R M F
      f : LinearPMap R E F
      z : M
      x_fst : E
      x_snd : F
      y : Subtype fun x => Membership.mem (HSMul.hSMul z f).domain x
      hy : Eq ↑y { fst := x_fst, snd := x_snd }.1
      h : Eq (HSMul.hSMul z (↑f y)) { fst := x_fst, snd := x_snd }.2
      ⊢ Exists fun a => Exists fun a_1 => And (Eq (↑a_1) a) (And (Eq a x_fst) (Eq (H …
    -/
    use x_fst, y, hy
    /-
      🎉 no goals
    -/
  /-
    case h.mk.mpr
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    h : Membership.mem (Submodule.map (LinearMap.id.prodMap (HSMul.hSMul z LinearM …
    ⊢ Membership.mem (HSMul.hSMul z f).graph { fst := x_fst, snd := x_snd }
  -/
  rw [Submodule.mem_map] at h
  /-
    case h.mk.mpr
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    h : Exists fun y => And (Membership.mem f.graph y) (Eq ((LinearMap.id.prodMap  …
    ⊢ Membership.mem (HSMul.hSMul z f).graph { fst := x_fst, snd := x_snd }
  -/
  rcases h with ⟨x', hx', h⟩
  /-
    case h.mk.mpr.intro.intro
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    x' : Prod E F
    hx' : Membership.mem f.graph x'
    h : Eq ((LinearMap.id.prodMap (HSMul.hSMul z LinearMap.id)) x') { fst := x_fst …
    ⊢ Membership.mem (HSMul.hSMul z f).graph { fst := x_fst, snd := x_snd }
  -/
  cases x'
  simp only [LinearMap.prodMap_apply, LinearMap.id_coe, id, LinearMap.smul_apply,
    Prod.mk.inj_iff] at h
  /-
    case h.mk.mpr.intro.intro.mk
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    hx' : Membership.mem f.graph { fst := fst✝, snd := snd✝ }
    h : And (Eq fst✝ x_fst) (Eq (HSMul.hSMul z snd✝) x_snd)
    ⊢ Membership.mem (HSMul.hSMul z f).graph { fst := x_fst, snd := x_snd }
  -/
  rw [mem_graph_iff] at hx' ⊢
  /-
    case h.mk.mpr.intro.intro.mk
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    hx' : Exists fun y => And (Eq ↑y { fst := fst✝, snd := snd✝ }.1) (Eq (↑f y) {  …
    h : And (Eq fst✝ x_fst) (Eq (HSMul.hSMul z snd✝) x_snd)
    ⊢ Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(HSMul.hS …
  -/
  rcases hx' with ⟨y, hy, hx'⟩
  /-
    case h.mk.mpr.intro.intro.mk.intro.intro
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    h : And (Eq fst✝ x_fst) (Eq (HSMul.hSMul z snd✝) x_snd)
    y : Subtype fun x => Membership.mem f.domain x
    hy : Eq ↑y { fst := fst✝, snd := snd✝ }.1
    hx' : Eq (↑f y) { fst := fst✝, snd := snd✝ }.2
    ⊢ Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(HSMul.hS …
  -/
  use y
  /-
    case h
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    h : And (Eq fst✝ x_fst) (Eq (HSMul.hSMul z snd✝) x_snd)
    y : Subtype fun x => Membership.mem f.domain x
    hy : Eq ↑y { fst := fst✝, snd := snd✝ }.1
    hx' : Eq (↑f y) { fst := fst✝, snd := snd✝ }.2
    ⊢ And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(HSMul.hSMul z f) y) { fs …
  -/
  rw [← h.1, ← h.2]
  /-
    case h
    R : Type u_1
    inst✝⁷ : Ring R
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module R E
    F : Type u_3
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module R F
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M F
    inst✝ : SMulCommClass R M F
    f : LinearPMap R E F
    z : M
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    h : And (Eq fst✝ x_fst) (Eq (HSMul.hSMul z snd✝) x_snd)
    y : Subtype fun x => Membership.mem f.domain x
    hy : Eq ↑y { fst := fst✝, snd := snd✝ }.1
    hx' : Eq (↑f y) { fst := fst✝, snd := snd✝ }.2
    ⊢ And (Eq ↑y { fst := fst✝, snd := HSMul.hSMul z snd✝ }.1) (Eq (↑(HSMul.hSMul  …
  -/
  simp [hy, hx']
  /-
    🎉 no goals
  -/


/-- The graph of `-f` as a pushforward. -/
theorem neg_graph (f : E →ₗ.[R] F) :
    (-f).graph =
    f.graph.map ((LinearMap.id : E →ₗ[R] E).prodMap (-(LinearMap.id : F →ₗ[R] F))) := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    ⊢ Eq (Neg.neg f).graph (Submodule.map (LinearMap.id.prodMap (Neg.neg LinearMap …
  -/
  ext x; cases' x with x_fst x_snd
  /-
    case h.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    ⊢ Iff (Membership.mem (Neg.neg f).graph { fst := x_fst, snd := x_snd }) (Membe …
  -/
  constructor <;> intro h
    /-
      case h.mk.mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x_fst : E
      x_snd : F
      h : Membership.mem (Neg.neg f).graph { fst := x_fst, snd := x_snd }
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (Neg.neg LinearMap.id))  …
    -/
  · rw [mem_graph_iff] at h
    /-
      case h.mk.mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x_fst : E
      x_snd : F
      h : Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(Neg.ne …
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (Neg.neg LinearMap.id))  …
    -/
    rcases h with ⟨y, hy, h⟩
    /-
      case h.mk.mp.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x_fst : E
      x_snd : F
      y : Subtype fun x => Membership.mem (Neg.neg f).domain x
      hy : Eq ↑y { fst := x_fst, snd := x_snd }.1
      h : Eq (↑(Neg.neg f) y) { fst := x_fst, snd := x_snd }.2
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (Neg.neg LinearMap.id))  …
    -/
    rw [LinearPMap.neg_apply] at h
    /-
      case h.mk.mp.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x_fst : E
      x_snd : F
      y : Subtype fun x => Membership.mem (Neg.neg f).domain x
      hy : Eq ↑y { fst := x_fst, snd := x_snd }.1
      h : Eq (Neg.neg (↑f y)) { fst := x_fst, snd := x_snd }.2
      ⊢ Membership.mem (Submodule.map (LinearMap.id.prodMap (Neg.neg LinearMap.id))  …
    -/
    rw [Submodule.mem_map]
    simp only [mem_graph_iff, LinearMap.prodMap_apply, LinearMap.id_coe, id,
      LinearMap.neg_apply, Prod.mk.inj_iff, Prod.exists, exists_exists_and_eq_and]
    /-
      case h.mk.mp.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x_fst : E
      x_snd : F
      y : Subtype fun x => Membership.mem (Neg.neg f).domain x
      hy : Eq ↑y { fst := x_fst, snd := x_snd }.1
      h : Eq (Neg.neg (↑f y)) { fst := x_fst, snd := x_snd }.2
      ⊢ Exists fun a => Exists fun a_1 => And (Eq (↑a_1) a) (And (Eq a x_fst) (Eq (N …
    -/
    use x_fst, y, hy
    /-
      🎉 no goals
    -/
  /-
    case h.mk.mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    h : Membership.mem (Submodule.map (LinearMap.id.prodMap (Neg.neg LinearMap.id) …
    ⊢ Membership.mem (Neg.neg f).graph { fst := x_fst, snd := x_snd }
  -/
  rw [Submodule.mem_map] at h
  /-
    case h.mk.mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    h : Exists fun y => And (Membership.mem f.graph y) (Eq ((LinearMap.id.prodMap  …
    ⊢ Membership.mem (Neg.neg f).graph { fst := x_fst, snd := x_snd }
  -/
  rcases h with ⟨x', hx', h⟩
  /-
    case h.mk.mpr.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    x' : Prod E F
    hx' : Membership.mem f.graph x'
    h : Eq ((LinearMap.id.prodMap (Neg.neg LinearMap.id)) x') { fst := x_fst, snd  …
    ⊢ Membership.mem (Neg.neg f).graph { fst := x_fst, snd := x_snd }
  -/
  cases x'
  simp only [LinearMap.prodMap_apply, LinearMap.id_coe, id, LinearMap.neg_apply,
    Prod.mk.inj_iff] at h
  /-
    case h.mk.mpr.intro.intro.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    hx' : Membership.mem f.graph { fst := fst✝, snd := snd✝ }
    h : And (Eq fst✝ x_fst) (Eq (Neg.neg snd✝) x_snd)
    ⊢ Membership.mem (Neg.neg f).graph { fst := x_fst, snd := x_snd }
  -/
  rw [mem_graph_iff] at hx' ⊢
  /-
    case h.mk.mpr.intro.intro.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    hx' : Exists fun y => And (Eq ↑y { fst := fst✝, snd := snd✝ }.1) (Eq (↑f y) {  …
    h : And (Eq fst✝ x_fst) (Eq (Neg.neg snd✝) x_snd)
    ⊢ Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(Neg.neg  …
  -/
  rcases hx' with ⟨y, hy, hx'⟩
  /-
    case h.mk.mpr.intro.intro.mk.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    h : And (Eq fst✝ x_fst) (Eq (Neg.neg snd✝) x_snd)
    y : Subtype fun x => Membership.mem f.domain x
    hy : Eq ↑y { fst := fst✝, snd := snd✝ }.1
    hx' : Eq (↑f y) { fst := fst✝, snd := snd✝ }.2
    ⊢ Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(Neg.neg  …
  -/
  use y
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    h : And (Eq fst✝ x_fst) (Eq (Neg.neg snd✝) x_snd)
    y : Subtype fun x => Membership.mem f.domain x
    hy : Eq ↑y { fst := fst✝, snd := snd✝ }.1
    hx' : Eq (↑f y) { fst := fst✝, snd := snd✝ }.2
    ⊢ And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑(Neg.neg f) y) { fst := x …
  -/
  rw [← h.1, ← h.2]
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x_fst : E
    x_snd : F
    fst✝ : E
    snd✝ : F
    h : And (Eq fst✝ x_fst) (Eq (Neg.neg snd✝) x_snd)
    y : Subtype fun x => Membership.mem f.domain x
    hy : Eq ↑y { fst := fst✝, snd := snd✝ }.1
    hx' : Eq (↑f y) { fst := fst✝, snd := snd✝ }.2
    ⊢ And (Eq ↑y { fst := fst✝, snd := Neg.neg snd✝ }.1) (Eq (↑(Neg.neg f) y) { fs …
  -/
  simp [hy, hx']
  /-
    🎉 no goals
  -/


theorem mem_graph_snd_inj (f : E →ₗ.[R] F) {x y : E} {x' y' : F} (hx : (x, x') ∈ f.graph)
    (hy : (y, y') ∈ f.graph) (hxy : x = y) : x' = y' := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : E
    x' y' : F
    hx : Membership.mem f.graph { fst := x, snd := x' }
    hy : Membership.mem f.graph { fst := y, snd := y' }
    hxy : Eq x y
    ⊢ Eq x' y'
  -/
  rw [mem_graph_iff] at hx hy
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : E
    x' y' : F
    hx : Exists fun y => And (Eq ↑y { fst := x, snd := x' }.1) (Eq (↑f y) { fst := …
    hy : Exists fun y_1 => And (Eq ↑y_1 { fst := y, snd := y' }.1) (Eq (↑f y_1) {  …
    hxy : Eq x y
    ⊢ Eq x' y'
  -/
  rcases hx with ⟨x'', hx1, hx2⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : E
    x' y' : F
    hy : Exists fun y_1 => And (Eq ↑y_1 { fst := y, snd := y' }.1) (Eq (↑f y_1) {  …
    hxy : Eq x y
    x'' : Subtype fun x => Membership.mem f.domain x
    hx1 : Eq ↑x'' { fst := x, snd := x' }.1
    hx2 : Eq (↑f x'') { fst := x, snd := x' }.2
    ⊢ Eq x' y'
  -/
  rcases hy with ⟨y'', hy1, hy2⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : E
    x' y' : F
    hxy : Eq x y
    x'' : Subtype fun x => Membership.mem f.domain x
    hx1 : Eq ↑x'' { fst := x, snd := x' }.1
    hx2 : Eq (↑f x'') { fst := x, snd := x' }.2
    y'' : Subtype fun x => Membership.mem f.domain x
    hy1 : Eq ↑y'' { fst := y, snd := y' }.1
    hy2 : Eq (↑f y'') { fst := y, snd := y' }.2
    ⊢ Eq x' y'
  -/
  simp only at hx1 hx2 hy1 hy2
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : E
    x' y' : F
    hxy : Eq x y
    x'' : Subtype fun x => Membership.mem f.domain x
    hx1 : Eq (↑x'') x
    hx2 : Eq (↑f x'') x'
    y'' : Subtype fun x => Membership.mem f.domain x
    hy1 : Eq (↑y'') y
    hy2 : Eq (↑f y'') y'
    ⊢ Eq x' y'
  -/
  rw [← hx1, ← hy1, SetLike.coe_eq_coe] at hxy
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : E
    x' y' : F
    x'' : Subtype fun x => Membership.mem f.domain x
    hx1 : Eq (↑x'') x
    hx2 : Eq (↑f x'') x'
    y'' : Subtype fun x => Membership.mem f.domain x
    hxy : Eq x'' y''
    hy1 : Eq (↑y'') y
    hy2 : Eq (↑f y'') y'
    ⊢ Eq x' y'
  -/
  rw [← hx2, ← hy2, hxy]
  /-
    🎉 no goals
  -/


theorem mem_graph_snd_inj' (f : E →ₗ.[R] F) {x y : E × F} (hx : x ∈ f.graph) (hy : y ∈ f.graph)
    (hxy : x.1 = y.1) : x.2 = y.2 := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x y : Prod E F
    hx : Membership.mem f.graph x
    hy : Membership.mem f.graph y
    hxy : Eq x.1 y.1
    ⊢ Eq x.2 y.2
  -/
  cases x
  /-
    case mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : Prod E F
    hy : Membership.mem f.graph y
    fst✝ : E
    snd✝ : F
    hx : Membership.mem f.graph { fst := fst✝, snd := snd✝ }
    hxy : Eq { fst := fst✝, snd := snd✝ }.1 y.1
    ⊢ Eq { fst := fst✝, snd := snd✝ }.2 y.2
  -/
  cases y
  /-
    case mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    fst✝¹ : E
    snd✝¹ : F
    hx : Membership.mem f.graph { fst := fst✝¹, snd := snd✝¹ }
    fst✝ : E
    snd✝ : F
    hy : Membership.mem f.graph { fst := fst✝, snd := snd✝ }
    hxy : Eq { fst := fst✝¹, snd := snd✝¹ }.1 { fst := fst✝, snd := snd✝ }.1
    ⊢ Eq { fst := fst✝¹, snd := snd✝¹ }.2 { fst := fst✝, snd := snd✝ }.2
  -/
  exact f.mem_graph_snd_inj hx hy hxy
  /-
    🎉 no goals
  -/


/-- The property that `f 0 = 0` in terms of the graph. -/
theorem graph_fst_eq_zero_snd (f : E →ₗ.[R] F) {x : E} {x' : F} (h : (x, x') ∈ f.graph)
    (hx : x = 0) : x' = 0 :=
  f.mem_graph_snd_inj h f.graph.zero_mem hx


theorem mem_domain_iff {f : E →ₗ.[R] F} {x : E} : x ∈ f.domain ↔ ∃ y : F, (x, y) ∈ f.graph := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    ⊢ Iff (Membership.mem f.domain x) (Exists fun y => Membership.mem f.graph { fs …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      h : Membership.mem f.domain x
      ⊢ Exists fun y => Membership.mem f.graph { fst := x, snd := y }
    -/
  · use f ⟨x, h⟩
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      h : Membership.mem f.domain x
      ⊢ Membership.mem f.graph { fst := x, snd := ↑f ⟨x, h⟩ }
    -/
    exact f.mem_graph ⟨x, h⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    h : Exists fun y => Membership.mem f.graph { fst := x, snd := y }
    ⊢ Membership.mem f.domain x
  -/
  cases' h with y h
  /-
    case mpr.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    h : Membership.mem f.graph { fst := x, snd := y }
    ⊢ Membership.mem f.domain x
  -/
  rw [mem_graph_iff] at h
  /-
    case mpr.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    h : Exists fun y_1 => And (Eq ↑y_1 { fst := x, snd := y }.1) (Eq (↑f y_1) { fs …
    ⊢ Membership.mem f.domain x
  -/
  cases' h with x' h
  /-
    case mpr.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    x' : Subtype fun x => Membership.mem f.domain x
    h : And (Eq ↑x' { fst := x, snd := y }.1) (Eq (↑f x') { fst := x, snd := y }.2)
    ⊢ Membership.mem f.domain x
  -/
  simp only at h
  /-
    case mpr.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    x' : Subtype fun x => Membership.mem f.domain x
    h : And (Eq (↑x') x) (Eq (↑f x') y)
    ⊢ Membership.mem f.domain x
  -/
  rw [← h.1]
  /-
    case mpr.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    x' : Subtype fun x => Membership.mem f.domain x
    h : And (Eq (↑x') x) (Eq (↑f x') y)
    ⊢ Membership.mem f.domain ↑x'
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_domain_of_mem_graph {f : E →ₗ.[R] F} {x : E} {y : F} (h : (x, y) ∈ f.graph) :
    x ∈ f.domain := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    h : Membership.mem f.graph { fst := x, snd := y }
    ⊢ Membership.mem f.domain x
  -/
  rw [mem_domain_iff]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    h : Membership.mem f.graph { fst := x, snd := y }
    ⊢ Exists fun y => Membership.mem f.graph { fst := x, snd := y }
  -/
  exact ⟨y, h⟩
  /-
    🎉 no goals
  -/


theorem image_iff {f : E →ₗ.[R] F} {x : E} {y : F} (hx : x ∈ f.domain) :
    y = f ⟨x, hx⟩ ↔ (x, y) ∈ f.graph := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    hx : Membership.mem f.domain x
    ⊢ Iff (Eq y (↑f ⟨x, hx⟩)) (Membership.mem f.graph { fst := x, snd := y })
  -/
  rw [mem_graph_iff]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    hx : Membership.mem f.domain x
    ⊢ Iff (Eq y (↑f ⟨x, hx⟩)) (Exists fun y_1 => And (Eq ↑y_1 { fst := x, snd := y …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      y : F
      hx : Membership.mem f.domain x
      h : Eq y (↑f ⟨x, hx⟩)
      ⊢ Exists fun y_1 => And (Eq ↑y_1 { fst := x, snd := y }.1) (Eq (↑f y_1) { fst  …
    -/
  · use ⟨x, hx⟩
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      x : E
      y : F
      hx : Membership.mem f.domain x
      h : Eq y (↑f ⟨x, hx⟩)
      ⊢ And (Eq ↑⟨x, hx⟩ { fst := x, snd := y }.1) (Eq (↑f ⟨x, hx⟩) { fst := x, snd  …
    -/
    simp [h]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    hx : Membership.mem f.domain x
    h : Exists fun y_1 => And (Eq ↑y_1 { fst := x, snd := y }.1) (Eq (↑f y_1) { fs …
    ⊢ Eq y (↑f ⟨x, hx⟩)
  -/
  rcases h with ⟨⟨x', hx'⟩, ⟨h1, h2⟩⟩
  /-
    case mpr.intro.mk.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    hx : Membership.mem f.domain x
    x' : E
    hx' : Membership.mem f.domain x'
    h1 : Eq ↑⟨x', hx'⟩ { fst := x, snd := y }.1
    h2 : Eq (↑f ⟨x', hx'⟩) { fst := x, snd := y }.2
    ⊢ Eq y (↑f ⟨x, hx⟩)
  -/
  simp only [Submodule.coe_mk] at h1 h2
  /-
    case mpr.intro.mk.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    x : E
    y : F
    hx : Membership.mem f.domain x
    x' : E
    hx' : Membership.mem f.domain x'
    h1 : Eq x' x
    h2 : Eq (↑f ⟨x', hx'⟩) y
    ⊢ Eq y (↑f ⟨x, hx⟩)
  -/
  simp only [← h2, h1]
  /-
    🎉 no goals
  -/


theorem mem_range_iff {f : E →ₗ.[R] F} {y : F} : y ∈ Set.range f ↔ ∃ x : E, (x, y) ∈ f.graph := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    ⊢ Iff (Membership.mem (Set.range ↑f) y) (Exists fun x => Membership.mem f.grap …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      y : F
      h : Membership.mem (Set.range ↑f) y
      ⊢ Exists fun x => Membership.mem f.graph { fst := x, snd := y }
    -/
  · rw [Set.mem_range] at h
    /-
      case mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      y : F
      h : Exists fun y_1 => Eq (↑f y_1) y
      ⊢ Exists fun x => Membership.mem f.graph { fst := x, snd := y }
    -/
    rcases h with ⟨⟨x, hx⟩, h⟩
    /-
      case mp.intro.mk
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      y : F
      x : E
      hx : Membership.mem f.domain x
      h : Eq (↑f ⟨x, hx⟩) y
      ⊢ Exists fun x => Membership.mem f.graph { fst := x, snd := y }
    -/
    use x
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      y : F
      x : E
      hx : Membership.mem f.domain x
      h : Eq (↑f ⟨x, hx⟩) y
      ⊢ Membership.mem f.graph { fst := x, snd := y }
    -/
    rw [← h]
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f : LinearPMap R E F
      y : F
      x : E
      hx : Membership.mem f.domain x
      h : Eq (↑f ⟨x, hx⟩) y
      ⊢ Membership.mem f.graph { fst := x, snd := ↑f ⟨x, hx⟩ }
    -/
    exact f.mem_graph ⟨x, hx⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    h : Exists fun x => Membership.mem f.graph { fst := x, snd := y }
    ⊢ Membership.mem (Set.range ↑f) y
  -/
  cases' h with x h
  /-
    case mpr.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    x : E
    h : Membership.mem f.graph { fst := x, snd := y }
    ⊢ Membership.mem (Set.range ↑f) y
  -/
  rw [mem_graph_iff] at h
  /-
    case mpr.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    x : E
    h : Exists fun y_1 => And (Eq ↑y_1 { fst := x, snd := y }.1) (Eq (↑f y_1) { fs …
    ⊢ Membership.mem (Set.range ↑f) y
  -/
  cases' h with x h
  /-
    case mpr.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    x✝ : E
    x : Subtype fun x => Membership.mem f.domain x
    h : And (Eq ↑x { fst := x✝, snd := y }.1) (Eq (↑f x) { fst := x✝, snd := y }.2)
    ⊢ Membership.mem (Set.range ↑f) y
  -/
  rw [Set.mem_range]
  /-
    case mpr.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    x✝ : E
    x : Subtype fun x => Membership.mem f.domain x
    h : And (Eq ↑x { fst := x✝, snd := y }.1) (Eq (↑f x) { fst := x✝, snd := y }.2)
    ⊢ Exists fun y_1 => Eq (↑f y_1) y
  -/
  use x
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    x✝ : E
    x : Subtype fun x => Membership.mem f.domain x
    h : And (Eq ↑x { fst := x✝, snd := y }.1) (Eq (↑f x) { fst := x✝, snd := y }.2)
    ⊢ Eq (↑f x) y
  -/
  simp only at h
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    y : F
    x✝ : E
    x : Subtype fun x => Membership.mem f.domain x
    h : And (Eq (↑x) x✝) (Eq (↑f x) y)
    ⊢ Eq (↑f x) y
  -/
  rw [h.2]
  /-
    🎉 no goals
  -/


theorem mem_domain_iff_of_eq_graph {f g : E →ₗ.[R] F} (h : f.graph = g.graph) {x : E} :
                                      /-
                                        R : Type u_1
                                        inst✝⁴ : Ring R
                                        E : Type u_2
                                        inst✝³ : AddCommGroup E
                                        inst✝² : Module R E
                                        F : Type u_3
                                        inst✝¹ : AddCommGroup F
                                        inst✝ : Module R F
                                        f g : LinearPMap R E F
                                        h : Eq f.graph g.graph
                                        x : E
                                        ⊢ Iff (Membership.mem f.domain x) (Membership.mem g.domain x)
                                      -/
    x ∈ f.domain ↔ x ∈ g.domain := by simp_rw [mem_domain_iff, h]
                                      /-
                                        🎉 no goals
                                      -/


theorem le_of_le_graph {f g : E →ₗ.[R] F} (h : f.graph ≤ g.graph) : f ≤ g := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    ⊢ LE.le f g
  -/
  constructor
    /-
      case left
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : LE.le f.graph g.graph
      ⊢ LE.le f.domain g.domain
    -/
  · intro x hx
    /-
      case left
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : LE.le f.graph g.graph
      x : E
      hx : Membership.mem f.domain x
      ⊢ Membership.mem g.domain x
    -/
    rw [mem_domain_iff] at hx ⊢
    /-
      case left
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : LE.le f.graph g.graph
      x : E
      hx : Exists fun y => Membership.mem f.graph { fst := x, snd := y }
      ⊢ Exists fun y => Membership.mem g.graph { fst := x, snd := y }
    -/
    cases' hx with y hx
    /-
      case left.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : LE.le f.graph g.graph
      x : E
      y : F
      hx : Membership.mem f.graph { fst := x, snd := y }
      ⊢ Exists fun y => Membership.mem g.graph { fst := x, snd := y }
    -/
    use y
    /-
      case h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : LE.le f.graph g.graph
      x : E
      y : F
      hx : Membership.mem f.graph { fst := x, snd := y }
      ⊢ Membership.mem g.graph { fst := x, snd := y }
    -/
    exact h hx
    /-
      🎉 no goals
    -/
  /-
    case right
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    ⊢ ∀ ⦃x : Subtype fun x => Membership.mem f.domain x⦄ ⦃y : Subtype fun x => Mem …
  -/
  rintro ⟨x, hx⟩ ⟨y, hy⟩ hxy
  /-
    case right.mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    x : E
    hx : Membership.mem f.domain x
    y : E
    hy : Membership.mem g.domain y
    hxy : Eq ↑⟨x, hx⟩ ↑⟨y, hy⟩
    ⊢ Eq (↑f ⟨x, hx⟩) (↑g ⟨y, hy⟩)
  -/
  rw [image_iff]
  /-
    case right.mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    x : E
    hx : Membership.mem f.domain x
    y : E
    hy : Membership.mem g.domain y
    hxy : Eq ↑⟨x, hx⟩ ↑⟨y, hy⟩
    ⊢ Membership.mem g.graph { fst := y, snd := ↑f ⟨x, hx⟩ }
  -/
  refine h ?_
  /-
    case right.mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    x : E
    hx : Membership.mem f.domain x
    y : E
    hy : Membership.mem g.domain y
    hxy : Eq ↑⟨x, hx⟩ ↑⟨y, hy⟩
    ⊢ Membership.mem f.graph { fst := y, snd := ↑f ⟨x, hx⟩ }
  -/
  simp only [Submodule.coe_mk] at hxy
  /-
    case right.mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    x : E
    hx : Membership.mem f.domain x
    y : E
    hy : Membership.mem g.domain y
    hxy : Eq x y
    ⊢ Membership.mem f.graph { fst := y, snd := ↑f ⟨x, hx⟩ }
  -/
  rw [hxy] at hx
  /-
    case right.mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    x : E
    hx✝ : Membership.mem f.domain x
    y : E
    hx : Membership.mem f.domain y
    hy : Membership.mem g.domain y
    hxy : Eq x y
    ⊢ Membership.mem f.graph { fst := y, snd := ↑f ⟨x, hx✝⟩ }
  -/
  rw [← image_iff hx]
  /-
    case right.mk.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f.graph g.graph
    x : E
    hx✝ : Membership.mem f.domain x
    y : E
    hx : Membership.mem f.domain y
    hy : Membership.mem g.domain y
    hxy : Eq x y
    ⊢ Eq (↑f ⟨x, hx✝⟩) (↑f ⟨y, hx⟩)
  -/
  simp [hxy]
  /-
    🎉 no goals
  -/


theorem le_graph_of_le {f g : E →ₗ.[R] F} (h : f ≤ g) : f.graph ≤ g.graph := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    ⊢ LE.le f.graph g.graph
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    hx : Membership.mem f.graph x
    ⊢ Membership.mem g.graph x
  -/
  rw [mem_graph_iff] at hx ⊢
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    hx : Exists fun y => And (Eq (↑y) x.1) (Eq (↑f y) x.2)
    ⊢ Exists fun y => And (Eq (↑y) x.1) (Eq (↑g y) x.2)
  -/
  cases' hx with y hx
  /-
    case intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    y : Subtype fun x => Membership.mem f.domain x
    hx : And (Eq (↑y) x.1) (Eq (↑f y) x.2)
    ⊢ Exists fun y => And (Eq (↑y) x.1) (Eq (↑g y) x.2)
  -/
  use ⟨y, h.1 y.2⟩
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    y : Subtype fun x => Membership.mem f.domain x
    hx : And (Eq (↑y) x.1) (Eq (↑f y) x.2)
    ⊢ And (Eq (↑⟨↑y, ⋯⟩) x.1) (Eq (↑g ⟨↑y, ⋯⟩) x.2)
  -/
  simp only [hx, Submodule.coe_mk, eq_self_iff_true, true_and]
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    y : Subtype fun x => Membership.mem f.domain x
    hx : And (Eq (↑y) x.1) (Eq (↑f y) x.2)
    ⊢ Eq (↑g ⟨x.1, ⋯⟩) x.2
  -/
  convert hx.2 using 1
  /-
    case h.e'_2
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    y : Subtype fun x => Membership.mem f.domain x
    hx : And (Eq (↑y) x.1) (Eq (↑f y) x.2)
    ⊢ Eq (↑g ⟨x.1, ⋯⟩) (↑f y)
  -/
  refine (h.2 ?_).symm
  /-
    case h.e'_2
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : LE.le f g
    x : Prod E F
    y : Subtype fun x => Membership.mem f.domain x
    hx : And (Eq (↑y) x.1) (Eq (↑f y) x.2)
    ⊢ Eq ↑y ↑⟨x.1, ⋯⟩
  -/
  simp only [hx.1, Submodule.coe_mk]
  /-
    🎉 no goals
  -/


theorem le_graph_iff {f g : E →ₗ.[R] F} : f.graph ≤ g.graph ↔ f ≤ g :=
  ⟨le_of_le_graph, le_graph_of_le⟩


theorem eq_of_eq_graph {f g : E →ₗ.[R] F} (h : f.graph = g.graph) : f = g := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f g : LinearPMap R E F
    h : Eq f.graph g.graph
    ⊢ Eq f g
  -/
  ext
    /-
      case h.h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : Eq f.graph g.graph
      x✝ : E
      ⊢ Iff (Membership.mem f.domain x✝) (Membership.mem g.domain x✝)
    -/
  · exact mem_domain_iff_of_eq_graph h
    /-
      🎉 no goals
    -/
    /-
      case h'
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : Eq f.graph g.graph
      x✝ : Subtype fun x => Membership.mem f.domain x
      y✝ : Subtype fun x => Membership.mem g.domain x
      _h✝ : Eq ↑x✝ ↑y✝
      ⊢ Eq (↑f x✝) (↑g y✝)
    -/
  · apply (le_of_le_graph h.le).2
    /-
      case h'._h
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      f g : LinearPMap R E F
      h : Eq f.graph g.graph
      x✝ : Subtype fun x => Membership.mem f.domain x
      y✝ : Subtype fun x => Membership.mem g.domain x
      _h✝ : Eq ↑x✝ ↑y✝
      ⊢ Eq ↑x✝ ↑y✝
    -/
    assumption
    /-
      🎉 no goals
    -/


theorem existsUnique_from_graph {g : Submodule R (E × F)}
    (hg : ∀ {x : E × F} (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0) {a : E}
    (ha : a ∈ g.map (LinearMap.fst R E F)) : ∃! b : F, (a, b) ∈ g := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ {x : Prod E F}, Membership.mem g x → Eq x.1 0 → Eq x.2 0
    a : E
    ha : Membership.mem (Submodule.map (LinearMap.fst R E F) g) a
    ⊢ ExistsUnique fun b => Membership.mem g { fst := a, snd := b }
  -/
  refine existsUnique_of_exists_of_unique ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      g : Submodule R (Prod E F)
      hg : ∀ {x : Prod E F}, Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : E
      ha : Membership.mem (Submodule.map (LinearMap.fst R E F) g) a
      ⊢ Exists fun x => Membership.mem g { fst := a, snd := x }
    -/
  · convert ha
    /-
      case a
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      g : Submodule R (Prod E F)
      hg : ∀ {x : Prod E F}, Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : E
      ha : Membership.mem (Submodule.map (LinearMap.fst R E F) g) a
      ⊢ Iff (Exists fun x => Membership.mem g { fst := a, snd := x }) (Membership.me …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ {x : Prod E F}, Membership.mem g x → Eq x.1 0 → Eq x.2 0
    a : E
    ha : Membership.mem (Submodule.map (LinearMap.fst R E F) g) a
    ⊢ ∀ (y₁ y₂ : F), Membership.mem g { fst := a, snd := y₁ } → Membership.mem g { …
  -/
  intro y₁ y₂ hy₁ hy₂
  have hy : ((0 : E), y₁ - y₂) ∈ g := by
    convert g.sub_mem hy₁ hy₂
    exact (sub_self _).symm
  /-
    case refine_2
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ {x : Prod E F}, Membership.mem g x → Eq x.1 0 → Eq x.2 0
    a : E
    ha : Membership.mem (Submodule.map (LinearMap.fst R E F) g) a
    y₁ y₂ : F
    hy₁ : Membership.mem g { fst := a, snd := y₁ }
    hy₂ : Membership.mem g { fst := a, snd := y₂ }
    hy : Membership.mem g { fst := 0, snd := HSub.hSub y₁ y₂ }
    ⊢ Eq y₁ y₂
  -/
  exact sub_eq_zero.mp (hg hy (by simp))
  /-
    🎉 no goals
  -/


/-- Auxiliary definition to unfold the existential quantifier. -/
noncomputable def valFromGraph {g : Submodule R (E × F)}
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0) {a : E}
    (ha : a ∈ g.map (LinearMap.fst R E F)) : F :=
  (ExistsUnique.exists (existsUnique_from_graph @hg ha)).choose


theorem valFromGraph_mem {g : Submodule R (E × F)}
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0) {a : E}
    (ha : a ∈ g.map (LinearMap.fst R E F)) : (a, valFromGraph hg ha) ∈ g :=
  (ExistsUnique.exists (existsUnique_from_graph @hg ha)).choose_spec


/-- Define a `LinearMap` from its graph.

Helper definition for `LinearPMap`. -/
noncomputable def toLinearPMapAux (g : Submodule R (E × F))
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0) :
    g.map (LinearMap.fst R E F) →ₗ[R] F where
  toFun := fun x => valFromGraph hg x.2
  map_add' := fun v w => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      v w : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      ⊢ Eq ((fun x => Submodule.valFromGraph hg ⋯) (HAdd.hAdd v w)) (HAdd.hAdd ((fun …
    -/
    have hadd := (g.map (LinearMap.fst R E F)).add_mem v.2 w.2
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      v w : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hadd : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HAdd.hAdd ↑v ↑w)
      ⊢ Eq ((fun x => Submodule.valFromGraph hg ⋯) (HAdd.hAdd v w)) (HAdd.hAdd ((fun …
    -/
    have hvw := valFromGraph_mem hg hadd
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      v w : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hadd : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HAdd.hAdd ↑v ↑w)
      hvw : Membership.mem g { fst := HAdd.hAdd ↑v ↑w, snd := Submodule.valFromGraph …
      ⊢ Eq ((fun x => Submodule.valFromGraph hg ⋯) (HAdd.hAdd v w)) (HAdd.hAdd ((fun …
    -/
    have hvw' := g.add_mem (valFromGraph_mem hg v.2) (valFromGraph_mem hg w.2)
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      v w : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hadd : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HAdd.hAdd ↑v ↑w)
      hvw : Membership.mem g { fst := HAdd.hAdd ↑v ↑w, snd := Submodule.valFromGraph …
      hvw' : Membership.mem g (HAdd.hAdd { fst := ↑v, snd := Submodule.valFromGraph  …
      ⊢ Eq ((fun x => Submodule.valFromGraph hg ⋯) (HAdd.hAdd v w)) (HAdd.hAdd ((fun …
    -/
    rw [Prod.mk_add_mk] at hvw'
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      v w : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hadd : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HAdd.hAdd ↑v ↑w)
      hvw : Membership.mem g { fst := HAdd.hAdd ↑v ↑w, snd := Submodule.valFromGraph …
      hvw' : Membership.mem g { fst := HAdd.hAdd ↑v ↑w, snd := HAdd.hAdd (Submodule. …
      ⊢ Eq ((fun x => Submodule.valFromGraph hg ⋯) (HAdd.hAdd v w)) (HAdd.hAdd ((fun …
    -/
    exact (existsUnique_from_graph @hg hadd).unique hvw hvw'
    /-
      🎉 no goals
    -/
  map_smul' := fun a v => by
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : R
      v : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      ⊢ Eq ({ toFun := fun x => Submodule.valFromGraph hg ⋯, map_add' := ⋯ }.toFun ( …
    -/
    have hsmul := (g.map (LinearMap.fst R E F)).smul_mem a v.2
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : R
      v : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hsmul : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HSMul.hSMul a  …
      ⊢ Eq ({ toFun := fun x => Submodule.valFromGraph hg ⋯, map_add' := ⋯ }.toFun ( …
    -/
    have hav := valFromGraph_mem hg hsmul
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : R
      v : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hsmul : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HSMul.hSMul a  …
      hav : Membership.mem g { fst := HSMul.hSMul a ↑v, snd := Submodule.valFromGrap …
      ⊢ Eq ({ toFun := fun x => Submodule.valFromGraph hg ⋯, map_add' := ⋯ }.toFun ( …
    -/
    have hav' := g.smul_mem a (valFromGraph_mem hg v.2)
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : R
      v : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hsmul : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HSMul.hSMul a  …
      hav : Membership.mem g { fst := HSMul.hSMul a ↑v, snd := Submodule.valFromGrap …
      hav' : Membership.mem g (HSMul.hSMul a { fst := ↑v, snd := Submodule.valFromGr …
      ⊢ Eq ({ toFun := fun x => Submodule.valFromGraph hg ⋯, map_add' := ⋯ }.toFun ( …
    -/
    rw [Prod.smul_mk] at hav'
    /-
      R : Type u_1
      inst✝⁶ : Ring R
      E : Type u_2
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module R E
      F : Type u_3
      inst✝³ : AddCommGroup F
      inst✝² : Module R F
      G : Type u_4
      inst✝¹ : AddCommGroup G
      inst✝ : Module R G
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      a : R
      v : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
      hsmul : Membership.mem (Submodule.map (LinearMap.fst R E F) g) (HSMul.hSMul a  …
      hav : Membership.mem g { fst := HSMul.hSMul a ↑v, snd := Submodule.valFromGrap …
      hav' : Membership.mem g { fst := HSMul.hSMul a ↑v, snd := HSMul.hSMul a (Submo …
      ⊢ Eq ({ toFun := fun x => Submodule.valFromGraph hg ⋯, map_add' := ⋯ }.toFun ( …
    -/
    exact (existsUnique_from_graph @hg hsmul).unique hav hav'
    /-
      🎉 no goals
    -/


open scoped Classical in
/-- Define a `LinearPMap` from its graph.

In the case that the submodule is not a graph of a `LinearPMap` then the underlying linear map
is just the zero map. -/
noncomputable def toLinearPMap (g : Submodule R (E × F)) : E →ₗ.[R] F where
  domain := g.map (LinearMap.fst R E F)
  toFun := if hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0 then
    g.toLinearPMapAux hg else 0


theorem toLinearPMap_domain (g : Submodule R (E × F)) :
    g.toLinearPMap.domain = g.map (LinearMap.fst R E F) := rfl


theorem toLinearPMap_apply_aux {g : Submodule R (E × F)}
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0)
    (x : g.map (LinearMap.fst R E F)) :
    g.toLinearPMap x = valFromGraph hg x.2 := by
  classical
  change (if hg : _ then g.toLinearPMapAux hg else 0) x = _
  rw [dif_pos]
  · rfl
  · exact hg


theorem mem_graph_toLinearPMap {g : Submodule R (E × F)}
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0)
    (x : g.map (LinearMap.fst R E F)) : (x.val, g.toLinearPMap x) ∈ g := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
    ⊢ Membership.mem g { fst := ↑x, snd := ↑g.toLinearPMap x }
  -/
  rw [toLinearPMap_apply_aux hg]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x : Subtype fun x => Membership.mem (Submodule.map (LinearMap.fst R E F) g) x
    ⊢ Membership.mem g { fst := ↑x, snd := Submodule.valFromGraph hg ⋯ }
  -/
  exact valFromGraph_mem hg x.2
  /-
    🎉 no goals
  -/


@[simp]
theorem toLinearPMap_graph_eq (g : Submodule R (E × F))
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0) :
    g.toLinearPMap.graph = g := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    ⊢ Eq g.toLinearPMap.graph g
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x : Prod E F
    ⊢ Iff (Membership.mem g.toLinearPMap.graph x) (Membership.mem g x)
  -/
  constructor <;> intro hx
    /-
      case h.mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      x : Prod E F
      hx : Membership.mem g.toLinearPMap.graph x
      ⊢ Membership.mem g x
    -/
  · rw [LinearPMap.mem_graph_iff] at hx
    /-
      case h.mp
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      x : Prod E F
      hx : Exists fun y => And (Eq (↑y) x.1) (Eq (↑g.toLinearPMap y) x.2)
      ⊢ Membership.mem g x
    -/
    rcases hx with ⟨y, hx1, hx2⟩
    /-
      case h.mp.intro.intro
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      x : Prod E F
      y : Subtype fun x => Membership.mem g.toLinearPMap.domain x
      hx1 : Eq (↑y) x.1
      hx2 : Eq (↑g.toLinearPMap y) x.2
      ⊢ Membership.mem g x
    -/
    convert g.mem_graph_toLinearPMap hg y using 1
    /-
      case h.e'_5
      R : Type u_1
      inst✝⁴ : Ring R
      E : Type u_2
      inst✝³ : AddCommGroup E
      inst✝² : Module R E
      F : Type u_3
      inst✝¹ : AddCommGroup F
      inst✝ : Module R F
      g : Submodule R (Prod E F)
      hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
      x : Prod E F
      y : Subtype fun x => Membership.mem g.toLinearPMap.domain x
      hx1 : Eq (↑y) x.1
      hx2 : Eq (↑g.toLinearPMap y) x.2
      ⊢ Eq x { fst := ↑y, snd := ↑g.toLinearPMap y }
    -/
    exact Prod.ext hx1.symm hx2.symm
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x : Prod E F
    hx : Membership.mem g x
    ⊢ Membership.mem g.toLinearPMap.graph x
  -/
  rw [LinearPMap.mem_graph_iff]
  /-
    case h.mpr
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x : Prod E F
    hx : Membership.mem g x
    ⊢ Exists fun y => And (Eq (↑y) x.1) (Eq (↑g.toLinearPMap y) x.2)
  -/
  cases' x with x_fst x_snd
  have hx_fst : x_fst ∈ g.map (LinearMap.fst R E F) := by
    simp only [mem_map, LinearMap.fst_apply, Prod.exists, exists_and_right, exists_eq_right]
    exact ⟨x_snd, hx⟩
  /-
    case h.mpr.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x_fst : E
    x_snd : F
    hx : Membership.mem g { fst := x_fst, snd := x_snd }
    hx_fst : Membership.mem (Submodule.map (LinearMap.fst R E F) g) x_fst
    ⊢ Exists fun y => And (Eq ↑y { fst := x_fst, snd := x_snd }.1) (Eq (↑g.toLinea …
  -/
  refine ⟨⟨x_fst, hx_fst⟩, Subtype.coe_mk x_fst hx_fst, ?_⟩
  /-
    case h.mpr.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x_fst : E
    x_snd : F
    hx : Membership.mem g { fst := x_fst, snd := x_snd }
    hx_fst : Membership.mem (Submodule.map (LinearMap.fst R E F) g) x_fst
    ⊢ Eq (↑g.toLinearPMap ⟨x_fst, hx_fst⟩) { fst := x_fst, snd := x_snd }.2
  -/
  rw [toLinearPMap_apply_aux hg]
  /-
    case h.mpr.mk
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    x_fst : E
    x_snd : F
    hx : Membership.mem g { fst := x_fst, snd := x_snd }
    hx_fst : Membership.mem (Submodule.map (LinearMap.fst R E F) g) x_fst
    ⊢ Eq (Submodule.valFromGraph hg ⋯) { fst := x_fst, snd := x_snd }.2
  -/
  exact (existsUnique_from_graph @hg hx_fst).unique (valFromGraph_mem hg hx_fst) hx
  /-
    🎉 no goals
  -/


theorem toLinearPMap_range (g : Submodule R (E × F))
    (hg : ∀ (x : E × F) (_hx : x ∈ g) (_hx' : x.fst = 0), x.snd = 0) :
    LinearMap.range g.toLinearPMap.toFun = g.map (LinearMap.snd R E F) := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    g : Submodule R (Prod E F)
    hg : ∀ (x : Prod E F), Membership.mem g x → Eq x.1 0 → Eq x.2 0
    ⊢ Eq (LinearMap.range g.toLinearPMap.toFun) (Submodule.map (LinearMap.snd R E  …
  -/
  rwa [← LinearPMap.graph_map_snd_eq_range, toLinearPMap_graph_eq]
  /-
    🎉 no goals
  -/


/-- The inverse of a `LinearPMap`. -/
noncomputable def inverse (f : E →ₗ.[R] F) : F →ₗ.[R] E :=
  (f.graph.map (LinearEquiv.prodComm R E F)).toLinearPMap


theorem inverse_domain : (inverse f).domain = LinearMap.range f.toFun := by
  rw [inverse, Submodule.toLinearPMap_domain, ← graph_map_snd_eq_range,
    ← LinearEquiv.fst_comp_prodComm, Submodule.map_comp]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    ⊢ Eq (Submodule.map (LinearMap.fst R F E) (Submodule.map (LinearEquiv.prodComm …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The graph of the inverse generates a `LinearPMap`. -/
theorem mem_inverse_graph_snd_eq_zero (x : F × E)
    (hv : x ∈ (graph f).map (LinearEquiv.prodComm R E F))
    (hv' : x.fst = 0) : x.snd = 0 := by
  simp only [Submodule.mem_map, mem_graph_iff, Subtype.exists, exists_and_left, exists_eq_left,
    LinearEquiv.prodComm_apply, Prod.exists, Prod.swap_prod_mk] at hv
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    x : Prod F E
    hv' : Eq x.1 0
    hv : Exists fun a => Exists fun b => And (Exists fun h => Eq (↑f ⟨a, ⋯⟩) b) (E …
    ⊢ Eq x.2 0
  -/
  rcases hv with ⟨a, b, ⟨ha, h1⟩, ⟨h2, h3⟩⟩
  /-
    case intro.intro.intro.intro.refl
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    a : E
    b : F
    ha : Membership.mem f.domain a
    h1 : Eq (↑f ⟨a, ⋯⟩) b
    hv' : Eq { fst := b, snd := a }.1 0
    ⊢ Eq { fst := b, snd := a }.2 0
  -/
  simp only at hv' ⊢
  /-
    case intro.intro.intro.intro.refl
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    a : E
    b : F
    ha : Membership.mem f.domain a
    h1 : Eq (↑f ⟨a, ⋯⟩) b
    hv' : Eq b 0
    ⊢ Eq a 0
  -/
  rw [hv'] at h1
  /-
    case intro.intro.intro.intro.refl
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    a : E
    b : F
    ha : Membership.mem f.domain a
    h1 : Eq (↑f ⟨a, ⋯⟩) 0
    hv' : Eq b 0
    ⊢ Eq a 0
  -/
  rw [LinearMap.ker_eq_bot'] at hf
  /-
    case intro.intro.intro.intro.refl
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : ∀ (m : Subtype fun x => Membership.mem f.domain x), Eq (f.toFun m) 0 → Eq …
    a : E
    b : F
    ha : Membership.mem f.domain a
    h1 : Eq (↑f ⟨a, ⋯⟩) 0
    hv' : Eq b 0
    ⊢ Eq a 0
  -/
  specialize hf ⟨a, ha⟩ h1
  /-
    case intro.intro.intro.intro.refl
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    a : E
    b : F
    ha : Membership.mem f.domain a
    h1 : Eq (↑f ⟨a, ⋯⟩) 0
    hv' : Eq b 0
    hf : Eq ⟨a, ha⟩ 0
    ⊢ Eq a 0
  -/
  simp only [Submodule.mk_eq_zero] at hf
  /-
    case intro.intro.intro.intro.refl
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    a : E
    b : F
    ha : Membership.mem f.domain a
    h1 : Eq (↑f ⟨a, ⋯⟩) 0
    hv' : Eq b 0
    hf : Eq a 0
    ⊢ Eq a 0
  -/
  exact hf
  /-
    🎉 no goals
  -/


theorem inverse_graph : (inverse f).graph = f.graph.map (LinearEquiv.prodComm R E F) := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    ⊢ Eq f.inverse.graph (Submodule.map (LinearEquiv.prodComm R E F) f.graph)
  -/
  rw [inverse, Submodule.toLinearPMap_graph_eq _ (mem_inverse_graph_snd_eq_zero hf)]
  /-
    🎉 no goals
  -/


theorem inverse_range : LinearMap.range (inverse f).toFun = f.domain := by
  rw [inverse, Submodule.toLinearPMap_range _ (mem_inverse_graph_snd_eq_zero hf),
    ← graph_map_fst_eq_domain, ← LinearEquiv.snd_comp_prodComm, Submodule.map_comp]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    ⊢ Eq (Submodule.map (LinearMap.snd R F E) (Submodule.map (LinearEquiv.prodComm …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_inverse_graph (x : f.domain) : (f x, (x : E)) ∈ (inverse f).graph := by
  simp only [inverse_graph hf, Submodule.mem_map, mem_graph_iff, Subtype.exists, exists_and_left,
    exists_eq_left, LinearEquiv.prodComm_apply, Prod.exists, Prod.swap_prod_mk, Prod.mk.injEq]
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    x : Subtype fun x => Membership.mem f.domain x
    ⊢ Exists fun a => Exists fun b => And (Exists fun h => Eq (↑f ⟨a, ⋯⟩) b) (And  …
  -/
  exact ⟨(x : E), f x, ⟨x.2, Eq.refl _⟩, Eq.refl _, Eq.refl _⟩
  /-
    🎉 no goals
  -/


theorem inverse_apply_eq {y : (inverse f).domain} {x : f.domain} (hxy : f x = y) :
    (inverse f) y = x := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    y : Subtype fun x => Membership.mem f.inverse.domain x
    x : Subtype fun x => Membership.mem f.domain x
    hxy : Eq (↑f x) ↑y
    ⊢ Eq (↑f.inverse y) ↑x
  -/
  have := mem_inverse_graph hf x
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    y : Subtype fun x => Membership.mem f.inverse.domain x
    x : Subtype fun x => Membership.mem f.domain x
    hxy : Eq (↑f x) ↑y
    this : Membership.mem f.inverse.graph { fst := ↑f x, snd := ↑x }
    ⊢ Eq (↑f.inverse y) ↑x
  -/
  simp only [mem_graph_iff, Subtype.exists, exists_and_left, exists_eq_left] at this
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    y : Subtype fun x => Membership.mem f.inverse.domain x
    x : Subtype fun x => Membership.mem f.domain x
    hxy : Eq (↑f x) ↑y
    this : Exists fun x_1 => Eq (↑f.inverse ⟨↑f x, ⋯⟩) ↑x
    ⊢ Eq (↑f.inverse y) ↑x
  -/
  rcases this with ⟨hx, h⟩
  /-
    case intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    y : Subtype fun x => Membership.mem f.inverse.domain x
    x : Subtype fun x => Membership.mem f.domain x
    hxy : Eq (↑f x) ↑y
    hx : Membership.mem f.inverse.domain (↑f x)
    h : Eq (↑f.inverse ⟨↑f x, ⋯⟩) ↑x
    ⊢ Eq (↑f.inverse y) ↑x
  -/
  rw [← h]
  /-
    case intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    y : Subtype fun x => Membership.mem f.inverse.domain x
    x : Subtype fun x => Membership.mem f.domain x
    hxy : Eq (↑f x) ↑y
    hx : Membership.mem f.inverse.domain (↑f x)
    h : Eq (↑f.inverse ⟨↑f x, ⋯⟩) ↑x
    ⊢ Eq (↑f.inverse y) (↑f.inverse ⟨↑f x, ⋯⟩)
  -/
  congr
  /-
    case intro.e_a
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    f : LinearPMap R E F
    hf : Eq (LinearMap.ker f.toFun) Bot.bot
    y : Subtype fun x => Membership.mem f.inverse.domain x
    x : Subtype fun x => Membership.mem f.domain x
    hxy : Eq (↑f x) ↑y
    hx : Membership.mem f.inverse.domain (↑f x)
    h : Eq (↑f.inverse ⟨↑f x, ⋯⟩) ↑x
    ⊢ Eq y ⟨↑f x, ⋯⟩
  -/
  simp only [hxy, Subtype.coe_eta]
  /-
    🎉 no goals
  -/


