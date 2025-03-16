/-- The kernel of a linear map `f : M → M₂` is defined to be `comap f ⊥`. This is equivalent to the
set of `x : M` such that `f x = 0`. The kernel is a submodule of `M`. -/
def ker (f : F) : Submodule R M :=
  comap f ⊥


@[simp]
theorem mem_ker {f : F} {y} : y ∈ ker f ↔ f y = 0 :=
  mem_bot R₂


@[simp]
theorem ker_id : ker (LinearMap.id : M →ₗ[R] M) = ⊥ :=
  rfl


@[simp]
theorem map_coe_ker (f : F) (x : ker f) : f x = 0 :=
  mem_ker.1 x.2


theorem ker_toAddSubmonoid (f : M →ₛₗ[τ₁₂] M₂) : f.ker.toAddSubmonoid = (AddMonoidHom.mker f) :=
  rfl


theorem comp_ker_subtype (f : M →ₛₗ[τ₁₂] M₂) : f.comp f.ker.subtype = 0 :=
  LinearMap.ext fun x => mem_ker.1 x.2


theorem ker_comp (f : M →ₛₗ[τ₁₂] M₂) (g : M₂ →ₛₗ[τ₂₃] M₃) :
    ker (g.comp f : M →ₛₗ[τ₁₃] M₃) = comap f (ker g) :=
  rfl


theorem ker_le_ker_comp (f : M →ₛₗ[τ₁₂] M₂) (g : M₂ →ₛₗ[τ₂₃] M₃) :
                                                 /-
                                                   R : Type u_1
                                                   R₂ : Type u_2
                                                   R₃ : Type u_3
                                                   M : Type u_5
                                                   M₂ : Type u_7
                                                   M₃ : Type u_8
                                                   inst✝⁹ : Semiring R
                                                   inst✝⁸ : Semiring R₂
                                                   inst✝⁷ : Semiring R₃
                                                   inst✝⁶ : AddCommMonoid M
                                                   inst✝⁵ : AddCommMonoid M₂
                                                   inst✝⁴ : AddCommMonoid M₃
                                                   inst✝³ : Module R M
                                                   inst✝² : Module R₂ M₂
                                                   inst✝¹ : Module R₃ M₃
                                                   τ₁₂ : RingHom R R₂
                                                   τ₂₃ : RingHom R₂ R₃
                                                   τ₁₃ : RingHom R R₃
                                                   inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
                                                   f : LinearMap τ₁₂ M M₂
                                                   g : LinearMap τ₂₃ M₂ M₃
                                                   ⊢ LE.le (LinearMap.ker f) (LinearMap.ker (g.comp f))
                                                 -/
    ker f ≤ ker (g.comp f : M →ₛₗ[τ₁₃] M₃) := by rw [ker_comp]; exact comap_mono bot_le
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem ker_sup_ker_le_ker_comp_of_commute {f g : M →ₗ[R] M} (h : Commute f g) :
    ker f ⊔ ker g ≤ ker (f ∘ₗ g) := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) M M
    h : Commute f g
    ⊢ LE.le (Max.max (LinearMap.ker f) (LinearMap.ker g)) (LinearMap.ker (f.comp g))
  -/
  refine sup_le_iff.mpr ⟨?_, ker_le_ker_comp g f⟩
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) M M
    h : Commute f g
    ⊢ LE.le (LinearMap.ker f) (LinearMap.ker (f.comp g))
  -/
  rw [← mul_eq_comp, h.eq, mul_eq_comp]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) M M
    h : Commute f g
    ⊢ LE.le (LinearMap.ker f) (LinearMap.ker (g.comp f))
  -/
  exact ker_le_ker_comp f g
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_le_comap {p : Submodule R₂ M₂} (f : M →ₛₗ[τ₁₂] M₂) :
    ker f ≤ p.comap f :=
                /-
                  R : Type u_1
                  R₂ : Type u_2
                  M : Type u_5
                  M₂ : Type u_7
                  inst✝⁵ : Semiring R
                  inst✝⁴ : Semiring R₂
                  inst✝³ : AddCommMonoid M
                  inst✝² : AddCommMonoid M₂
                  inst✝¹ : Module R M
                  inst✝ : Module R₂ M₂
                  τ₁₂ : RingHom R R₂
                  p : Submodule R₂ M₂
                  f : LinearMap τ₁₂ M M₂
                  x : M
                  hx : Membership.mem (LinearMap.ker f) x
                  ⊢ Membership.mem (Submodule.comap f p) x
                -/
  fun x hx ↦ by simp [mem_ker.mp hx]
                /-
                  🎉 no goals
                -/


theorem disjoint_ker {f : F} {p : Submodule R M} :
    Disjoint p (ker f) ↔ ∀ x ∈ p, f x = 0 → x = 0 := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_11
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    p : Submodule R M
    ⊢ Iff (Disjoint p (LinearMap.ker f)) (∀ (x : M), Membership.mem p x → Eq (f x) …
  -/
  simp [disjoint_def]
  /-
    🎉 no goals
  -/


theorem ker_eq_bot' {f : F} : ker f = ⊥ ↔ ∀ m, f m = 0 → m = 0 := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_11
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    ⊢ Iff (Eq (LinearMap.ker f) Bot.bot) (∀ (m : M), Eq (f m) 0 → Eq m 0)
  -/
  simpa [disjoint_iff_inf_le] using disjoint_ker (f := f) (p := ⊤)
  /-
    🎉 no goals
  -/


theorem ker_eq_bot_of_inverse {τ₂₁ : R₂ →+* R} [RingHomInvPair τ₁₂ τ₂₁] {f : M →ₛₗ[τ₁₂] M₂}
    {g : M₂ →ₛₗ[τ₂₁] M} (h : (g.comp f : M →ₗ[R] M) = id) : ker f = ⊥ :=
                               /-
                                 R : Type u_1
                                 R₂ : Type u_2
                                 M : Type u_5
                                 M₂ : Type u_7
                                 inst✝⁶ : Semiring R
                                 inst✝⁵ : Semiring R₂
                                 inst✝⁴ : AddCommMonoid M
                                 inst✝³ : AddCommMonoid M₂
                                 inst✝² : Module R M
                                 inst✝¹ : Module R₂ M₂
                                 τ₁₂ : RingHom R R₂
                                 τ₂₁ : RingHom R₂ R
                                 inst✝ : RingHomInvPair τ₁₂ τ₂₁
                                 f : LinearMap τ₁₂ M M₂
                                 g : LinearMap τ₂₁ M₂ M
                                 h : Eq (g.comp f) LinearMap.id
                                 m : M
                                 hm : Eq (f m) 0
                                 ⊢ Eq m 0
                               -/
  ker_eq_bot'.2 fun m hm => by rw [← id_apply (R := R) m, ← h, comp_apply, hm, g.map_zero]
                               /-
                                 🎉 no goals
                               -/


theorem le_ker_iff_map [RingHomSurjective τ₁₂] {f : F} {p : Submodule R M} :
                                  /-
                                    R : Type u_1
                                    R₂ : Type u_2
                                    M : Type u_5
                                    M₂ : Type u_7
                                    inst✝⁸ : Semiring R
                                    inst✝⁷ : Semiring R₂
                                    inst✝⁶ : AddCommMonoid M
                                    inst✝⁵ : AddCommMonoid M₂
                                    inst✝⁴ : Module R M
                                    inst✝³ : Module R₂ M₂
                                    τ₁₂ : RingHom R R₂
                                    F : Type u_11
                                    inst✝² : FunLike F M M₂
                                    inst✝¹ : SemilinearMapClass F τ₁₂ M M₂
                                    inst✝ : RingHomSurjective τ₁₂
                                    f : F
                                    p : Submodule R M
                                    ⊢ Iff (LE.le p (LinearMap.ker f)) (Eq (Submodule.map f p) Bot.bot)
                                  -/
    p ≤ ker f ↔ map f p = ⊥ := by rw [ker, eq_bot_iff, map_le_iff_le_comap]
                                  /-
                                    🎉 no goals
                                  -/


theorem ker_codRestrict {τ₂₁ : R₂ →+* R} (p : Submodule R M) (f : M₂ →ₛₗ[τ₂₁] M) (hf) :
                                           /-
                                             R : Type u_1
                                             R₂ : Type u_2
                                             M : Type u_5
                                             M₂ : Type u_7
                                             inst✝⁵ : Semiring R
                                             inst✝⁴ : Semiring R₂
                                             inst✝³ : AddCommMonoid M
                                             inst✝² : AddCommMonoid M₂
                                             inst✝¹ : Module R M
                                             inst✝ : Module R₂ M₂
                                             τ₂₁ : RingHom R₂ R
                                             p : Submodule R M
                                             f : LinearMap τ₂₁ M₂ M
                                             hf : ∀ (c : M₂), Membership.mem p (f c)
                                             ⊢ Eq (LinearMap.ker (LinearMap.codRestrict p f hf)) (LinearMap.ker f)
                                           -/
    ker (codRestrict p f hf) = ker f := by rw [ker, comap_codRestrict, Submodule.map_bot]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


lemma ker_domRestrict [AddCommMonoid M₁] [Module R M₁] (p : Submodule R M) (f : M →ₗ[R] M₁) :
    ker (domRestrict f p) = (ker f).comap p.subtype := ker_comp ..


theorem ker_restrict [AddCommMonoid M₁] [Module R M₁] {p : Submodule R M} {q : Submodule R M₁}
    {f : M →ₗ[R] M₁} (hf : ∀ x : M, x ∈ p → f x ∈ q) :
    ker (f.restrict hf) = (ker f).comap p.subtype := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R M₁
    p : Submodule R M
    q : Submodule R M₁
    f : LinearMap (RingHom.id R) M M₁
    hf : ∀ (x : M), Membership.mem p x → Membership.mem q (f x)
    ⊢ Eq (LinearMap.ker (f.restrict hf)) (Submodule.comap p.subtype (LinearMap.ker …
  -/
  rw [restrict_eq_codRestrict_domRestrict, ker_codRestrict, ker_domRestrict]
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_zero : ker (0 : M →ₛₗ[τ₁₂] M₂) = ⊤ :=
                            /-
                              R : Type u_1
                              R₂ : Type u_2
                              M : Type u_5
                              M₂ : Type u_7
                              inst✝⁵ : Semiring R
                              inst✝⁴ : Semiring R₂
                              inst✝³ : AddCommMonoid M
                              inst✝² : AddCommMonoid M₂
                              inst✝¹ : Module R M
                              inst✝ : Module R₂ M₂
                              τ₁₂ : RingHom R R₂
                              x : M
                              ⊢ Membership.mem (LinearMap.ker 0) x
                            -/
  eq_top_iff'.2 fun x => by simp
                            /-
                              🎉 no goals
                            -/


theorem ker_eq_top {f : M →ₛₗ[τ₁₂] M₂} : ker f = ⊤ ↔ f = 0 :=
  ⟨fun h => ext fun _ => mem_ker.1 <| h.symm ▸ trivial, fun h => h.symm ▸ ker_zero⟩


@[simp]
theorem _root_.AddMonoidHom.coe_toIntLinearMap_ker {M M₂ : Type*} [AddCommGroup M] [AddCommGroup M₂]
    (f : M →+ M₂) : LinearMap.ker f.toIntLinearMap = AddSubgroup.toIntSubmodule f.ker := rfl


theorem ker_eq_bot_of_injective {f : F} (hf : Injective f) : ker f = ⊥ := by
  have : Disjoint ⊤ (ker f) := by
    -- Porting note: `← map_zero f` should work here, but it needs to be directly applied to H.
    rw [disjoint_ker]
    intros _ _ H
    rw [← map_zero f] at H
    exact hf H
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_11
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    hf : Function.Injective ⇑f
    this : Disjoint Top.top (LinearMap.ker f)
    ⊢ Eq (LinearMap.ker f) Bot.bot
  -/
  simpa [disjoint_iff_inf_le]
  /-
    🎉 no goals
  -/


/-- The increasing sequence of submodules consisting of the kernels of the iterates of a linear map.
-/
@[simps]
def iterateKer (f : M →ₗ[R] M) : ℕ →o Submodule R M where
  toFun n := ker (f ^ n)
  monotone' n m w x h := by
    /-
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      V : Type u_9
      V₂ : Type u_10
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : Semiring R₂
      inst✝⁹ : Semiring R₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : Module R M
      inst✝⁴ : Module R₂ M₂
      inst✝³ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝² : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      F : Type u_11
      inst✝¹ : FunLike F M M₂
      inst✝ : SemilinearMapClass F τ₁₂ M M₂
      f : LinearMap (RingHom.id R) M M
      n m : Nat
      w : LE.le n m
      x : M
      h : Membership.mem ((fun n => LinearMap.ker (HPow.hPow f n)) n) x
      ⊢ Membership.mem ((fun n => LinearMap.ker (HPow.hPow f n)) m) x
    -/
    obtain ⟨c, rfl⟩ := Nat.exists_eq_add_of_le w
    /-
      case intro
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      V : Type u_9
      V₂ : Type u_10
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : Semiring R₂
      inst✝⁹ : Semiring R₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : Module R M
      inst✝⁴ : Module R₂ M₂
      inst✝³ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝² : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      F : Type u_11
      inst✝¹ : FunLike F M M₂
      inst✝ : SemilinearMapClass F τ₁₂ M M₂
      f : LinearMap (RingHom.id R) M M
      n : Nat
      x : M
      h : Membership.mem ((fun n => LinearMap.ker (HPow.hPow f n)) n) x
      c : Nat
      w : LE.le n (HAdd.hAdd n c)
      ⊢ Membership.mem ((fun n => LinearMap.ker (HPow.hPow f n)) (HAdd.hAdd n c)) x
    -/
    rw [LinearMap.mem_ker] at h
    /-
      case intro
      R : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      K : Type u_4
      M : Type u_5
      M₁ : Type u_6
      M₂ : Type u_7
      M₃ : Type u_8
      V : Type u_9
      V₂ : Type u_10
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : Semiring R₂
      inst✝⁹ : Semiring R₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : Module R M
      inst✝⁴ : Module R₂ M₂
      inst✝³ : Module R₃ M₃
      τ₁₂ : RingHom R R₂
      τ₂₃ : RingHom R₂ R₃
      τ₁₃ : RingHom R R₃
      inst✝² : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
      F : Type u_11
      inst✝¹ : FunLike F M M₂
      inst✝ : SemilinearMapClass F τ₁₂ M M₂
      f : LinearMap (RingHom.id R) M M
      n : Nat
      x : M
      h : Eq ((HPow.hPow f n) x) 0
      c : Nat
      w : LE.le n (HAdd.hAdd n c)
      ⊢ Membership.mem ((fun n => LinearMap.ker (HPow.hPow f n)) (HAdd.hAdd n c)) x
    -/
    rw [LinearMap.mem_ker, add_comm, pow_add, LinearMap.mul_apply, h, LinearMap.map_zero]
    /-
      🎉 no goals
    -/


theorem ker_toAddSubgroup (f : M →ₛₗ[τ₁₂] M₂) : (ker f).toAddSubgroup = f.toAddMonoidHom.ker :=
  rfl


                                                                /-
                                                                  R : Type u_1
                                                                  R₂ : Type u_2
                                                                  M : Type u_5
                                                                  M₂ : Type u_7
                                                                  inst✝⁷ : Ring R
                                                                  inst✝⁶ : Ring R₂
                                                                  inst✝⁵ : AddCommGroup M
                                                                  inst✝⁴ : AddCommGroup M₂
                                                                  inst✝³ : Module R M
                                                                  inst✝² : Module R₂ M₂
                                                                  τ₁₂ : RingHom R R₂
                                                                  F : Type u_11
                                                                  inst✝¹ : FunLike F M M₂
                                                                  inst✝ : SemilinearMapClass F τ₁₂ M M₂
                                                                  f : F
                                                                  x y : M
                                                                  ⊢ Iff (Membership.mem (LinearMap.ker f) (HSub.hSub x y)) (Eq (f x) (f y))
                                                                -/
theorem sub_mem_ker_iff {x y} : x - y ∈ ker f ↔ f x = f y := by rw [mem_ker, map_sub, sub_eq_zero]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem disjoint_ker' {p : Submodule R M} :
    Disjoint p (ker f) ↔ ∀ x ∈ p, ∀ y ∈ p, f x = f y → x = y :=
  disjoint_ker.trans
                                                                       /-
                                                                         R : Type u_1
                                                                         R₂ : Type u_2
                                                                         M : Type u_5
                                                                         M₂ : Type u_7
                                                                         inst✝⁷ : Ring R
                                                                         inst✝⁶ : Ring R₂
                                                                         inst✝⁵ : AddCommGroup M
                                                                         inst✝⁴ : AddCommGroup M₂
                                                                         inst✝³ : Module R M
                                                                         inst✝² : Module R₂ M₂
                                                                         τ₁₂ : RingHom R R₂
                                                                         F : Type u_11
                                                                         inst✝¹ : FunLike F M M₂
                                                                         inst✝ : SemilinearMapClass F τ₁₂ M M₂
                                                                         f : F
                                                                         p : Submodule R M
                                                                         H : ∀ (x : M), Membership.mem p x → Eq (f x) 0 → Eq x 0
                                                                         x : M
                                                                         hx : Membership.mem p x
                                                                         y : M
                                                                         hy : Membership.mem p y
                                                                         h : Eq (f x) (f y)
                                                                         ⊢ Eq (f (HSub.hSub x y)) 0
                                                                       -/
    ⟨fun H x hx y hy h => eq_of_sub_eq_zero <| H _ (sub_mem hx hy) (by simp [h]),
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                /-
                                                  R : Type u_1
                                                  R₂ : Type u_2
                                                  M : Type u_5
                                                  M₂ : Type u_7
                                                  inst✝⁷ : Ring R
                                                  inst✝⁶ : Ring R₂
                                                  inst✝⁵ : AddCommGroup M
                                                  inst✝⁴ : AddCommGroup M₂
                                                  inst✝³ : Module R M
                                                  inst✝² : Module R₂ M₂
                                                  τ₁₂ : RingHom R R₂
                                                  F : Type u_11
                                                  inst✝¹ : FunLike F M M₂
                                                  inst✝ : SemilinearMapClass F τ₁₂ M M₂
                                                  f : F
                                                  p : Submodule R M
                                                  H : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem p y → Eq (f x) ( …
                                                  x : M
                                                  h₁ : Membership.mem p x
                                                  h₂ : Eq (f x) 0
                                                  ⊢ Eq (f x) (f 0)
                                                -/
     fun H x h₁ h₂ => H x h₁ 0 (zero_mem _) (by simpa using h₂)⟩
                                                /-
                                                  🎉 no goals
                                                -/


theorem injOn_of_disjoint_ker {p : Submodule R M} {s : Set M} (h : s ⊆ p)
    (hd : Disjoint p (ker f)) : Set.InjOn f s := fun _ hx _ hy =>
  disjoint_ker'.1 hd _ (h hx) _ (h hy)


theorem _root_.LinearMapClass.ker_eq_bot : ker f = ⊥ ↔ Injective f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_7
    inst✝⁷ : Ring R
    inst✝⁶ : Ring R₂
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_11
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    ⊢ Iff (Eq (LinearMap.ker f) Bot.bot) (Function.Injective ⇑f)
  -/
  simpa [disjoint_iff_inf_le] using disjoint_ker' (f := f) (p := ⊤)
  /-
    🎉 no goals
  -/


theorem ker_eq_bot {f : M →ₛₗ[τ₁₂] M₂} : ker f = ⊥ ↔ Injective f :=
  LinearMapClass.ker_eq_bot _


@[simp] lemma injective_domRestrict_iff {f : M →ₛₗ[τ₁₂] M₂} {S : Submodule R M} :
    Injective (f.domRestrict S) ↔ S ⊓ LinearMap.ker f = ⊥ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Ring R
    inst✝⁴ : Ring R₂
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    ⊢ Iff (Function.Injective ⇑(f.domRestrict S)) (Eq (Min.min S (LinearMap.ker f) …
  -/
  rw [← LinearMap.ker_eq_bot]
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Ring R
    inst✝⁴ : Ring R₂
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    ⊢ Iff (Eq (LinearMap.ker (f.domRestrict S)) Bot.bot) (Eq (Min.min S (LinearMap …
  -/
  refine ⟨fun h ↦ le_bot_iff.1 ?_, fun h ↦ le_bot_iff.1 ?_⟩
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.ker (f.domRestrict S)) Bot.bot
      ⊢ LE.le (Min.min S (LinearMap.ker f)) Bot.bot
    -/
  · intro x ⟨hx, h'x⟩
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.ker (f.domRestrict S)) Bot.bot
      x : M
      hx : Membership.mem (↑S) x
      h'x : Membership.mem (↑(LinearMap.ker f)) x
      ⊢ Membership.mem Bot.bot x
    -/
    have : ⟨x, hx⟩ ∈ LinearMap.ker (LinearMap.domRestrict f S) := by simpa using h'x
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.ker (f.domRestrict S)) Bot.bot
      x : M
      hx : Membership.mem (↑S) x
      h'x : Membership.mem (↑(LinearMap.ker f)) x
      this : Membership.mem (LinearMap.ker (f.domRestrict S)) ⟨x, hx⟩
      ⊢ Membership.mem Bot.bot x
    -/
    rw [h] at this
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.ker (f.domRestrict S)) Bot.bot
      x : M
      hx : Membership.mem (↑S) x
      h'x : Membership.mem (↑(LinearMap.ker f)) x
      this : Membership.mem Bot.bot ⟨x, hx⟩
      ⊢ Membership.mem Bot.bot x
    -/
    simpa [mk_eq_zero] using this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Min.min S (LinearMap.ker f)) Bot.bot
      ⊢ LE.le (LinearMap.ker (f.domRestrict S)) Bot.bot
    -/
  · rintro ⟨x, hx⟩ h'x
    /-
      case refine_2.mk
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Min.min S (LinearMap.ker f)) Bot.bot
      x : M
      hx : Membership.mem S x
      h'x : Membership.mem (LinearMap.ker (f.domRestrict S)) ⟨x, hx⟩
      ⊢ Membership.mem Bot.bot ⟨x, hx⟩
    -/
    have : x ∈ S ⊓ LinearMap.ker f := ⟨hx, h'x⟩
    /-
      case refine_2.mk
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Min.min S (LinearMap.ker f)) Bot.bot
      x : M
      hx : Membership.mem S x
      h'x : Membership.mem (LinearMap.ker (f.domRestrict S)) ⟨x, hx⟩
      this : Membership.mem (Min.min S (LinearMap.ker f)) x
      ⊢ Membership.mem Bot.bot ⟨x, hx⟩
    -/
    rw [h] at this
    /-
      case refine_2.mk
      R : Type u_1
      R₂ : Type u_2
      M : Type u_5
      M₂ : Type u_7
      inst✝⁵ : Ring R
      inst✝⁴ : Ring R₂
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R M
      inst✝ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Min.min S (LinearMap.ker f)) Bot.bot
      x : M
      hx : Membership.mem S x
      h'x : Membership.mem (LinearMap.ker (f.domRestrict S)) ⟨x, hx⟩
      this : Membership.mem Bot.bot x
      ⊢ Membership.mem Bot.bot ⟨x, hx⟩
    -/
    simpa [mk_eq_zero] using this
    /-
      🎉 no goals
    -/


@[simp] theorem injective_restrict_iff_disjoint {p : Submodule R M} {f : M →ₗ[R] M}
    (hf : ∀ x ∈ p, f x ∈ p) :
    Injective (f.restrict hf) ↔ Disjoint p (ker f) := by
  rw [← ker_eq_bot, ker_restrict hf, ← ker_domRestrict, ker_eq_bot, injective_domRestrict_iff,
    disjoint_iff]


theorem ker_smul (f : V →ₗ[K] V₂) (a : K) (h : a ≠ 0) : ker (a • f) = ker f :=
  Submodule.comap_smul f _ a h


theorem ker_smul' (f : V →ₗ[K] V₂) (a : K) : ker (a • f) = ⨅ _ : a ≠ 0, ker f :=
  Submodule.comap_smul' f _ a


@[simp]
theorem comap_bot (f : F) : comap f ⊥ = ker f :=
  rfl


@[simp]
theorem ker_subtype : ker p.subtype = ⊥ :=
  ker_eq_bot_of_injective fun _ _ => Subtype.ext_val


@[simp]
theorem ker_inclusion (p p' : Submodule R M) (h : p ≤ p') : ker (inclusion h) = ⊥ := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p p' : Submodule R M
    h : LE.le p p'
    ⊢ Eq (LinearMap.ker (Submodule.inclusion h)) Bot.bot
  -/
  rw [inclusion, ker_codRestrict, ker_subtype]
  /-
    🎉 no goals
  -/


theorem ker_comp_of_ker_eq_bot (f : M →ₛₗ[τ₁₂] M₂) {g : M₂ →ₛₗ[τ₂₃] M₃} (hg : ker g = ⊥) :
                                                 /-
                                                   R : Type u_1
                                                   R₂ : Type u_2
                                                   R₃ : Type u_3
                                                   M : Type u_5
                                                   M₂ : Type u_7
                                                   M₃ : Type u_8
                                                   inst✝⁹ : Semiring R
                                                   inst✝⁸ : Semiring R₂
                                                   inst✝⁷ : Semiring R₃
                                                   inst✝⁶ : AddCommMonoid M
                                                   inst✝⁵ : AddCommMonoid M₂
                                                   inst✝⁴ : AddCommMonoid M₃
                                                   inst✝³ : Module R M
                                                   inst✝² : Module R₂ M₂
                                                   inst✝¹ : Module R₃ M₃
                                                   τ₁₂ : RingHom R R₂
                                                   τ₂₃ : RingHom R₂ R₃
                                                   τ₁₃ : RingHom R R₃
                                                   inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
                                                   f : LinearMap τ₁₂ M M₂
                                                   g : LinearMap τ₂₃ M₂ M₃
                                                   hg : Eq (LinearMap.ker g) Bot.bot
                                                   ⊢ Eq (LinearMap.ker (g.comp f)) (LinearMap.ker f)
                                                 -/
    ker (g.comp f : M →ₛₗ[τ₁₃] M₃) = ker f := by rw [ker_comp, hg, Submodule.comap_bot]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
protected theorem ker : LinearMap.ker (e : M →ₛₗ[σ₁₂] M₂) = ⊥ :=
  LinearMap.ker_eq_bot_of_injective e.toEquiv.injective


@[simp]
theorem ker_comp (l : M →ₛₗ[σ₁₂] M₂) :
    LinearMap.ker (((e'' : M₂ →ₛₗ[σ₂₃] M₃).comp l : M →ₛₗ[σ₁₃] M₃) : M →ₛₗ[σ₁₃] M₃) =
    LinearMap.ker l :=
  LinearMap.ker_comp_of_ker_eq_bot _ e''.ker


