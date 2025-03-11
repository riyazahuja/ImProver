/-- We say that `f` is `Θ(g)` along a filter `l` (notation: `f =Θ[l] g`) if `f =O[l] g` and
`g =O[l] f`. -/
def IsTheta (l : Filter α) (f : α → E) (g : α → F) : Prop :=
  IsBigO l f g ∧ IsBigO l g f


@[inherit_doc]
notation:100 f " =Θ[" l "] " g:100 => IsTheta l f g


theorem IsBigO.antisymm (h₁ : f =O[l] g) (h₂ : g =O[l] f) : f =Θ[l] g :=
  ⟨h₁, h₂⟩


lemma IsTheta.isBigO (h : f =Θ[l] g) : f =O[l] g := h.1


lemma IsTheta.isBigO_symm (h : f =Θ[l] g) : g =O[l] f := h.2


@[refl]
theorem isTheta_refl (f : α → E) (l : Filter α) : f =Θ[l] f :=
  ⟨isBigO_refl _ _, isBigO_refl _ _⟩


theorem isTheta_rfl : f =Θ[l] f :=
  isTheta_refl _ _


@[symm]
nonrec theorem IsTheta.symm (h : f =Θ[l] g) : g =Θ[l] f :=
  h.symm


theorem isTheta_comm : f =Θ[l] g ↔ g =Θ[l] f :=
  ⟨fun h ↦ h.symm, fun h ↦ h.symm⟩


@[trans]
theorem IsTheta.trans {f : α → E} {g : α → F'} {k : α → G} (h₁ : f =Θ[l] g) (h₂ : g =Θ[l] k) :
    f =Θ[l] k :=
  ⟨h₁.1.trans h₂.1, h₂.2.trans h₁.2⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → F') (γ := α → G) (IsTheta l) (IsTheta l) (IsTheta l) :=
  ⟨IsTheta.trans⟩


@[trans]
theorem IsBigO.trans_isTheta {f : α → E} {g : α → F'} {k : α → G} (h₁ : f =O[l] g)
    (h₂ : g =Θ[l] k) : f =O[l] k :=
  h₁.trans h₂.1

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → F') (γ := α → G) (IsBigO l) (IsTheta l) (IsBigO l) :=
  ⟨IsBigO.trans_isTheta⟩


@[trans]
theorem IsTheta.trans_isBigO {f : α → E} {g : α → F'} {k : α → G} (h₁ : f =Θ[l] g)
    (h₂ : g =O[l] k) : f =O[l] k :=
  h₁.1.trans h₂

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → F') (γ := α → G) (IsTheta l) (IsBigO l) (IsBigO l) :=
  ⟨IsTheta.trans_isBigO⟩


@[trans]
theorem IsLittleO.trans_isTheta {f : α → E} {g : α → F} {k : α → G'} (h₁ : f =o[l] g)
    (h₂ : g =Θ[l] k) : f =o[l] k :=
  h₁.trans_isBigO h₂.1

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → F') (γ := α → G') (IsLittleO l) (IsTheta l) (IsLittleO l) :=
  ⟨IsLittleO.trans_isTheta⟩


@[trans]
theorem IsTheta.trans_isLittleO {f : α → E} {g : α → F'} {k : α → G} (h₁ : f =Θ[l] g)
    (h₂ : g =o[l] k) : f =o[l] k :=
  h₁.1.trans_isLittleO h₂

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → F') (γ := α → G) (IsTheta l) (IsLittleO l) (IsLittleO l) :=
  ⟨IsTheta.trans_isLittleO⟩


@[trans]
theorem IsTheta.trans_eventuallyEq {f : α → E} {g₁ g₂ : α → F} (h : f =Θ[l] g₁) (hg : g₁ =ᶠ[l] g₂) :
    f =Θ[l] g₂ :=
  ⟨h.1.trans_eventuallyEq hg, hg.symm.trans_isBigO h.2⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → F) (γ := α → F) (IsTheta l) (EventuallyEq l) (IsTheta l) :=
  ⟨IsTheta.trans_eventuallyEq⟩


@[trans]
theorem _root_.Filter.EventuallyEq.trans_isTheta {f₁ f₂ : α → E} {g : α → F} (hf : f₁ =ᶠ[l] f₂)
    (h : f₂ =Θ[l] g) : f₁ =Θ[l] g :=
  ⟨hf.trans_isBigO h.1, h.2.trans_eventuallyEq hf.symm⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (α := α → E) (β := α → E) (γ := α → F) (EventuallyEq l) (IsTheta l) (IsTheta l) :=
  ⟨EventuallyEq.trans_isTheta⟩


lemma _root_.Filter.EventuallyEq.isTheta {f g : α → E} (h : f =ᶠ[l] g) : f =Θ[l] g :=
  h.trans_isTheta isTheta_rfl


@[simp]
                                      /-
                                        α : Type u_1
                                        E : Type u_3
                                        F : Type u_4
                                        inst✝¹ : Norm E
                                        inst✝ : Norm F
                                        f : α → E
                                        g : α → F
                                        ⊢ Asymptotics.IsTheta Bot.bot f g
                                      -/
theorem isTheta_bot : f =Θ[⊥] g := by simp [IsTheta]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
                                                                        /-
                                                                          α : Type u_1
                                                                          F : Type u_4
                                                                          E' : Type u_6
                                                                          inst✝¹ : Norm F
                                                                          inst✝ : SeminormedAddCommGroup E'
                                                                          g : α → F
                                                                          f' : α → E'
                                                                          l : Filter α
                                                                          ⊢ Iff (Asymptotics.IsTheta l (fun x => Norm.norm (f' x)) g) (Asymptotics.IsThe …
                                                                        -/
theorem isTheta_norm_left : (fun x ↦ ‖f' x‖) =Θ[l] g ↔ f' =Θ[l] g := by simp [IsTheta]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
                                                                         /-
                                                                           α : Type u_1
                                                                           E : Type u_3
                                                                           F' : Type u_7
                                                                           inst✝¹ : Norm E
                                                                           inst✝ : SeminormedAddCommGroup F'
                                                                           f : α → E
                                                                           g' : α → F'
                                                                           l : Filter α
                                                                           ⊢ Iff (Asymptotics.IsTheta l f fun x => Norm.norm (g' x)) (Asymptotics.IsTheta …
                                                                         -/
theorem isTheta_norm_right : (f =Θ[l] fun x ↦ ‖g' x‖) ↔ f =Θ[l] g' := by simp [IsTheta]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


alias ⟨IsTheta.of_norm_left, IsTheta.norm_left⟩ := isTheta_norm_left


alias ⟨IsTheta.of_norm_right, IsTheta.norm_right⟩ := isTheta_norm_right


theorem isTheta_of_norm_eventuallyEq (h : (fun x ↦ ‖f x‖) =ᶠ[l] fun x ↦ ‖g x‖) : f =Θ[l] g :=
                           /-
                             α : Type u_1
                             E : Type u_3
                             F : Type u_4
                             inst✝¹ : Norm E
                             inst✝ : Norm F
                             f : α → E
                             g : α → F
                             l : Filter α
                             h : l.EventuallyEq (fun x => Norm.norm (f x)) fun x => Norm.norm (g x)
                             ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul 1 (Norm.norm  …
                           -/
  ⟨IsBigO.of_bound 1 <| by simpa only [one_mul] using h.le,
                           /-
                             🎉 no goals
                           -/
                            /-
                              α : Type u_1
                              E : Type u_3
                              F : Type u_4
                              inst✝¹ : Norm E
                              inst✝ : Norm F
                              f : α → E
                              g : α → F
                              l : Filter α
                              h : l.EventuallyEq (fun x => Norm.norm (f x)) fun x => Norm.norm (g x)
                              ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (HMul.hMul 1 (Norm.norm  …
                            -/
    IsBigO.of_bound 1 <| by simpa only [one_mul] using h.symm.le⟩
                            /-
                              🎉 no goals
                            -/


theorem isTheta_of_norm_eventuallyEq' {g : α → ℝ} (h : (fun x ↦ ‖f' x‖) =ᶠ[l] g) : f' =Θ[l] g :=
                                                       /-
                                                         α : Type u_1
                                                         E' : Type u_6
                                                         inst✝ : SeminormedAddCommGroup E'
                                                         f' : α → E'
                                                         l : Filter α
                                                         g : α → Real
                                                         h : l.EventuallyEq (fun x => Norm.norm (f' x)) g
                                                         x : α
                                                         hx : Eq ((fun x => Norm.norm (f' x)) x) (g x)
                                                         ⊢ Eq ((fun x => Norm.norm (f' x)) x) ((fun x => Norm.norm (g x)) x)
                                                       -/
  isTheta_of_norm_eventuallyEq <| h.mono fun x hx ↦ by simp only [← hx, norm_norm]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem IsTheta.isLittleO_congr_left (h : f' =Θ[l] g') : f' =o[l] k ↔ g' =o[l] k :=
  ⟨h.symm.trans_isLittleO, h.trans_isLittleO⟩


theorem IsTheta.isLittleO_congr_right (h : g' =Θ[l] k') : f =o[l] g' ↔ f =o[l] k' :=
  ⟨fun H ↦ H.trans_isTheta h, fun H ↦ H.trans_isTheta h.symm⟩


theorem IsTheta.isBigO_congr_left (h : f' =Θ[l] g') : f' =O[l] k ↔ g' =O[l] k :=
  ⟨h.symm.trans_isBigO, h.trans_isBigO⟩


theorem IsTheta.isBigO_congr_right (h : g' =Θ[l] k') : f =O[l] g' ↔ f =O[l] k' :=
  ⟨fun H ↦ H.trans_isTheta h, fun H ↦ H.trans_isTheta h.symm⟩


lemma IsTheta.isTheta_congr_left (h : f' =Θ[l] g') : f' =Θ[l] k ↔ g' =Θ[l] k :=
  h.isBigO_congr_left.and h.isBigO_congr_right


lemma IsTheta.isTheta_congr_right (h : f' =Θ[l] g') : k =Θ[l] f' ↔ k =Θ[l] g' :=
  h.isBigO_congr_right.and h.isBigO_congr_left


theorem IsTheta.mono (h : f =Θ[l] g) (hl : l' ≤ l) : f =Θ[l'] g :=
  ⟨h.1.mono hl, h.2.mono hl⟩


theorem IsTheta.sup (h : f' =Θ[l] g') (h' : f' =Θ[l'] g') : f' =Θ[l ⊔ l'] g' :=
  ⟨h.1.sup h'.1, h.2.sup h'.2⟩


@[simp]
theorem isTheta_sup : f' =Θ[l ⊔ l'] g' ↔ f' =Θ[l] g' ∧ f' =Θ[l'] g' :=
  ⟨fun h ↦ ⟨h.mono le_sup_left, h.mono le_sup_right⟩, fun h ↦ h.1.sup h.2⟩


theorem IsTheta.eq_zero_iff (h : f'' =Θ[l] g'') : ∀ᶠ x in l, f'' x = 0 ↔ g'' x = 0 :=
  h.1.eq_zero_imp.mp <| h.2.eq_zero_imp.mono fun _ ↦ Iff.intro


theorem IsTheta.tendsto_zero_iff (h : f'' =Θ[l] g'') :
    Tendsto f'' l (𝓝 0) ↔ Tendsto g'' l (𝓝 0) := by
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f'' : α → E''
    g'' : α → F''
    l : Filter α
    h : Asymptotics.IsTheta l f'' g''
    ⊢ Iff (Filter.Tendsto f'' l (nhds 0)) (Filter.Tendsto g'' l (nhds 0))
  -/
  simp only [← isLittleO_one_iff ℝ, h.isLittleO_congr_left]
  /-
    🎉 no goals
  -/


theorem IsTheta.tendsto_norm_atTop_iff (h : f' =Θ[l] g') :
    Tendsto (norm ∘ f') l atTop ↔ Tendsto (norm ∘ g') l atTop := by
  simp only [Function.comp_def, ← isLittleO_const_left_of_ne (one_ne_zero' ℝ),
    h.isLittleO_congr_right]


theorem IsTheta.isBoundedUnder_le_iff (h : f' =Θ[l] g') :
    IsBoundedUnder (· ≤ ·) l (norm ∘ f') ↔ IsBoundedUnder (· ≤ ·) l (norm ∘ g') := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : SeminormedAddCommGroup F'
    f' : α → E'
    g' : α → F'
    l : Filter α
    h : Asymptotics.IsTheta l f' g'
    ⊢ Iff (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm. …
  -/
  simp only [← isBigO_const_of_ne (one_ne_zero' ℝ), h.isBigO_congr_left]
  /-
    🎉 no goals
  -/


theorem IsTheta.smul [NormedSpace 𝕜 E'] [NormedSpace 𝕜' F'] {f₁ : α → 𝕜} {f₂ : α → 𝕜'} {g₁ : α → E'}
    {g₂ : α → F'} (hf : f₁ =Θ[l] f₂) (hg : g₁ =Θ[l] g₂) :
    (fun x ↦ f₁ x • g₁ x) =Θ[l] fun x ↦ f₂ x • g₂ x :=
  ⟨hf.1.smul hg.1, hf.2.smul hg.2⟩


theorem IsTheta.mul {f₁ f₂ : α → 𝕜} {g₁ g₂ : α → 𝕜'} (h₁ : f₁ =Θ[l] g₁) (h₂ : f₂ =Θ[l] g₂) :
    (fun x ↦ f₁ x * f₂ x) =Θ[l] fun x ↦ g₁ x * g₂ x :=
  h₁.smul h₂


theorem IsTheta.inv {f : α → 𝕜} {g : α → 𝕜'} (h : f =Θ[l] g) :
    (fun x ↦ (f x)⁻¹) =Θ[l] fun x ↦ (g x)⁻¹ :=
  ⟨h.2.inv_rev h.1.eq_zero_imp, h.1.inv_rev h.2.eq_zero_imp⟩


@[simp]
theorem isTheta_inv {f : α → 𝕜} {g : α → 𝕜'} :
    ((fun x ↦ (f x)⁻¹) =Θ[l] fun x ↦ (g x)⁻¹) ↔ f =Θ[l] g :=
              /-
                α : Type u_1
                𝕜 : Type u_14
                𝕜' : Type u_15
                inst✝¹ : NormedField 𝕜
                inst✝ : NormedField 𝕜'
                l : Filter α
                f : α → 𝕜
                g : α → 𝕜'
                h : Asymptotics.IsTheta l (fun x => Inv.inv (f x)) fun x => Inv.inv (g x)
                ⊢ Asymptotics.IsTheta l f g
              -/
  ⟨fun h ↦ by simpa only [inv_inv] using h.inv, IsTheta.inv⟩
              /-
                🎉 no goals
              -/


theorem IsTheta.div {f₁ f₂ : α → 𝕜} {g₁ g₂ : α → 𝕜'} (h₁ : f₁ =Θ[l] g₁) (h₂ : f₂ =Θ[l] g₂) :
    (fun x ↦ f₁ x / f₂ x) =Θ[l] fun x ↦ g₁ x / g₂ x := by
  /-
    α : Type u_1
    𝕜 : Type u_14
    𝕜' : Type u_15
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedField 𝕜'
    l : Filter α
    f₁ f₂ : α → 𝕜
    g₁ g₂ : α → 𝕜'
    h₁ : Asymptotics.IsTheta l f₁ g₁
    h₂ : Asymptotics.IsTheta l f₂ g₂
    ⊢ Asymptotics.IsTheta l (fun x => HDiv.hDiv (f₁ x) (f₂ x)) fun x => HDiv.hDiv  …
  -/
  simpa only [div_eq_mul_inv] using h₁.mul h₂.inv
  /-
    🎉 no goals
  -/


theorem IsTheta.pow {f : α → 𝕜} {g : α → 𝕜'} (h : f =Θ[l] g) (n : ℕ) :
    (fun x ↦ f x ^ n) =Θ[l] fun x ↦ g x ^ n :=
  ⟨h.1.pow n, h.2.pow n⟩


theorem IsTheta.zpow {f : α → 𝕜} {g : α → 𝕜'} (h : f =Θ[l] g) (n : ℤ) :
    (fun x ↦ f x ^ n) =Θ[l] fun x ↦ g x ^ n := by
  /-
    α : Type u_1
    𝕜 : Type u_14
    𝕜' : Type u_15
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedField 𝕜'
    l : Filter α
    f : α → 𝕜
    g : α → 𝕜'
    h : Asymptotics.IsTheta l f g
    n : Int
    ⊢ Asymptotics.IsTheta l (fun x => HPow.hPow (f x) n) fun x => HPow.hPow (g x) n
  -/
  cases n
    /-
      case ofNat
      α : Type u_1
      𝕜 : Type u_14
      𝕜' : Type u_15
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedField 𝕜'
      l : Filter α
      f : α → 𝕜
      g : α → 𝕜'
      h : Asymptotics.IsTheta l f g
      a✝ : Nat
      ⊢ Asymptotics.IsTheta l (fun x => HPow.hPow (f x) (Int.ofNat a✝)) fun x => HPo …
    -/
  · simpa only [Int.ofNat_eq_coe, zpow_natCast] using h.pow _
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      α : Type u_1
      𝕜 : Type u_14
      𝕜' : Type u_15
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedField 𝕜'
      l : Filter α
      f : α → 𝕜
      g : α → 𝕜'
      h : Asymptotics.IsTheta l f g
      a✝ : Nat
      ⊢ Asymptotics.IsTheta l (fun x => HPow.hPow (f x) (Int.negSucc a✝)) fun x => H …
    -/
  · simpa only [zpow_negSucc] using (h.pow _).inv
    /-
      🎉 no goals
    -/


theorem isTheta_const_const {c₁ : E''} {c₂ : F''} (h₁ : c₁ ≠ 0) (h₂ : c₂ ≠ 0) :
    (fun _ : α ↦ c₁) =Θ[l] fun _ ↦ c₂ :=
  ⟨isBigO_const_const _ h₂ _, isBigO_const_const _ h₁ _⟩


@[simp]
theorem isTheta_const_const_iff [NeBot l] {c₁ : E''} {c₂ : F''} :
    ((fun _ : α ↦ c₁) =Θ[l] fun _ ↦ c₂) ↔ (c₁ = 0 ↔ c₂ = 0) := by
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝² : NormedAddCommGroup E''
    inst✝¹ : NormedAddCommGroup F''
    l : Filter α
    inst✝ : l.NeBot
    c₁ : E''
    c₂ : F''
    ⊢ Iff (Asymptotics.IsTheta l (fun x => c₁) fun x => c₂) (Iff (Eq c₁ 0) (Eq c₂  …
  -/
  simpa only [IsTheta, isBigO_const_const_iff, ← iff_def] using Iff.comm
  /-
    🎉 no goals
  -/


@[simp]
theorem isTheta_zero_left : (fun _ ↦ (0 : E')) =Θ[l] g'' ↔ g'' =ᶠ[l] 0 := by
  /-
    α : Type u_1
    E' : Type u_6
    F'' : Type u_10
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : NormedAddCommGroup F''
    g'' : α → F''
    l : Filter α
    ⊢ Iff (Asymptotics.IsTheta l (fun x => 0) g'') (l.EventuallyEq g'' 0)
  -/
  simp only [IsTheta, isBigO_zero, isBigO_zero_right_iff, true_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem isTheta_zero_right : (f'' =Θ[l] fun _ ↦ (0 : F')) ↔ f'' =ᶠ[l] 0 :=
  isTheta_comm.trans isTheta_zero_left


theorem isTheta_const_smul_left [NormedSpace 𝕜 E'] {c : 𝕜} (hc : c ≠ 0) :
    (fun x ↦ c • f' x) =Θ[l] g ↔ f' =Θ[l] g :=
  and_congr (isBigO_const_smul_left hc) (isBigO_const_smul_right hc)


alias ⟨IsTheta.of_const_smul_left, IsTheta.const_smul_left⟩ := isTheta_const_smul_left


theorem isTheta_const_smul_right [NormedSpace 𝕜 F'] {c : 𝕜} (hc : c ≠ 0) :
    (f =Θ[l] fun x ↦ c • g' x) ↔ f =Θ[l] g' :=
  and_congr (isBigO_const_smul_right hc) (isBigO_const_smul_left hc)


alias ⟨IsTheta.of_const_smul_right, IsTheta.const_smul_right⟩ := isTheta_const_smul_right


theorem isTheta_const_mul_left {c : 𝕜} {f : α → 𝕜} (hc : c ≠ 0) :
    (fun x ↦ c * f x) =Θ[l] g ↔ f =Θ[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    𝕜 : Type u_14
    inst✝¹ : Norm F
    inst✝ : NormedField 𝕜
    g : α → F
    l : Filter α
    c : 𝕜
    f : α → 𝕜
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsTheta l (fun x => HMul.hMul c (f x)) g) (Asymptotics.IsTh …
  -/
  simpa only [← smul_eq_mul] using isTheta_const_smul_left hc
  /-
    🎉 no goals
  -/


alias ⟨IsTheta.of_const_mul_left, IsTheta.const_mul_left⟩ := isTheta_const_mul_left


theorem isTheta_const_mul_right {c : 𝕜} {g : α → 𝕜} (hc : c ≠ 0) :
    (f =Θ[l] fun x ↦ c * g x) ↔ f =Θ[l] g := by
  /-
    α : Type u_1
    E : Type u_3
    𝕜 : Type u_14
    inst✝¹ : Norm E
    inst✝ : NormedField 𝕜
    f : α → E
    l : Filter α
    c : 𝕜
    g : α → 𝕜
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsTheta l f fun x => HMul.hMul c (g x)) (Asymptotics.IsThet …
  -/
  simpa only [← smul_eq_mul] using isTheta_const_smul_right hc
  /-
    🎉 no goals
  -/


alias ⟨IsTheta.of_const_mul_right, IsTheta.const_mul_right⟩ := isTheta_const_mul_right


theorem IsLittleO.right_isTheta_add {f₁ f₂ : α → E'} (h : f₁ =o[l] f₂) :
    f₂ =Θ[l] (f₁ + f₂) :=
  ⟨h.right_isBigO_add, h.add_isBigO (isBigO_refl _ _)⟩


theorem IsLittleO.right_isTheta_add' {f₁ f₂ : α → E'} (h : f₁ =o[l] f₂) :
    f₂ =Θ[l] (f₂ + f₁) :=
  add_comm f₁ f₂ ▸ h.right_isTheta_add


lemma IsTheta.add_isLittleO {f₁ f₂ : α → E'} {g : α → F}
    (hΘ : f₁ =Θ[l] g) (ho : f₂ =o[l] g) : (f₁ + f₂) =Θ[l] g :=
  (ho.trans_isTheta hΘ.symm).right_isTheta_add'.symm.trans hΘ


lemma IsLittleO.add_isTheta {f₁ f₂ : α → E'} {g : α → F}
    (ho : f₁ =o[l] g) (hΘ : f₂ =Θ[l] g) : (f₁ + f₂) =Θ[l] g :=
  add_comm f₁ f₂ ▸ hΘ.add_isLittleO ho


protected theorem IsTheta.fiberwise_right :
    f =Θ[l ×ˢ l'] g → ∀ᶠ x in l, (f ⟨x, ·⟩) =Θ[l'] (g ⟨x, ·⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ Asymptotics.IsTheta (SProd.sprod l l') f g → Filter.Eventually (fun x => Asy …
  -/
  simp only [IsTheta, eventually_and]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ And (Asymptotics.IsBigO (SProd.sprod l l') f g) (Asymptotics.IsBigO (SProd.s …
  -/
  exact fun ⟨h₁, h₂⟩ ↦ ⟨h₁.fiberwise_right, h₂.fiberwise_right⟩
  /-
    🎉 no goals
  -/


protected theorem IsTheta.fiberwise_left :
    f =Θ[l ×ˢ l'] g → ∀ᶠ y in l', (f ⟨·, y⟩) =Θ[l] (g ⟨·, y⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ Asymptotics.IsTheta (SProd.sprod l l') f g → Filter.Eventually (fun y => Asy …
  -/
  simp only [IsTheta, eventually_and]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ And (Asymptotics.IsBigO (SProd.sprod l l') f g) (Asymptotics.IsBigO (SProd.s …
  -/
  exact fun ⟨h₁, h₂⟩ ↦ ⟨h₁.fiberwise_left, h₂.fiberwise_left⟩
  /-
    🎉 no goals
  -/


protected theorem IsTheta.comp_fst : f =Θ[l] g → (f ∘ Prod.fst) =Θ[l ×ˢ l'] (g ∘ Prod.fst) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ Asymptotics.IsTheta l f g → Asymptotics.IsTheta (SProd.sprod l l') (Function …
  -/
  simp only [IsTheta, eventually_and]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ And (Asymptotics.IsBigO l f g) (Asymptotics.IsBigO l g f) → And (Asymptotics …
  -/
  exact fun ⟨h₁, h₂⟩ ↦ ⟨h₁.comp_fst l', h₂.comp_fst l'⟩
  /-
    🎉 no goals
  -/


protected theorem IsTheta.comp_snd : f =Θ[l] g → (f ∘ Prod.snd) =Θ[l' ×ˢ l] (g ∘ Prod.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ Asymptotics.IsTheta l f g → Asymptotics.IsTheta (SProd.sprod l' l) (Function …
  -/
  simp only [IsTheta, eventually_and]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ And (Asymptotics.IsBigO l f g) (Asymptotics.IsBigO l g f) → And (Asymptotics …
  -/
  exact fun ⟨h₁, h₂⟩ ↦ ⟨h₁.comp_snd l', h₂.comp_snd l'⟩
  /-
    🎉 no goals
  -/


protected theorem isTheta_principal
    (hf : ContinuousOn f s) (hs : IsCompact s) (hc : ‖c‖ ≠ 0) (hC : ∀ i ∈ s, f i ≠ 0) :
    f =Θ[𝓟 s] fun _ => c :=
  ⟨hf.isBigO_principal hs hc, hf.isBigO_rev_principal hs hC c⟩


