/-- Multilinear maps over the ring `R`, from `∀ i, M₁ i` to `M₂` where `M₁ i` and `M₂` are modules
over `R`. -/
structure MultilinearMap (R : Type uR) {ι : Type uι} (M₁ : ι → Type v₁) (M₂ : Type v₂) [Semiring R]
  [∀ i, AddCommMonoid (M₁ i)] [AddCommMonoid M₂] [∀ i, Module R (M₁ i)] [Module R M₂] where
  /-- The underlying multivariate function of a multilinear map. -/
  toFun : (∀ i, M₁ i) → M₂
  /-- A multilinear map is additive in every argument. -/
  map_update_add' :
    ∀ [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (x y : M₁ i),
      toFun (update m i (x + y)) = toFun (update m i x) + toFun (update m i y)
  /-- A multilinear map is compatible with scalar multiplication in every argument. -/
  map_update_smul' :
    ∀ [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (c : R) (x : M₁ i),
      toFun (update m i (c • x)) = c • toFun (update m i x)

-- Porting note: added to avoid a linter timeout.

instance : FunLike (MultilinearMap R M₁ M₂) (∀ i, M₁ i) M₂ where
  coe f := f.toFun
                             /-
                               R : Type uR
                               S : Type uS
                               ι : Type uι
                               n : Nat
                               M : Fin n.succ → Type v
                               M₁ : ι → Type v₁
                               M₂ : Type v₂
                               M₃ : Type v₃
                               M' : Type v'
                               inst✝¹⁰ : Semiring R
                               inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
                               inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                               inst✝⁷ : AddCommMonoid M₂
                               inst✝⁶ : AddCommMonoid M₃
                               inst✝⁵ : AddCommMonoid M'
                               inst✝⁴ : (i : Fin n.succ) → Module R (M i)
                               inst✝³ : (i : ι) → Module R (M₁ i)
                               inst✝² : Module R M₂
                               inst✝¹ : Module R M₃
                               inst✝ : Module R M'
                               f✝ f' f g : MultilinearMap R M₁ M₂
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; cases h; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem toFun_eq_coe : f.toFun = ⇑f :=
  rfl


@[simp]
theorem coe_mk (f : (∀ i, M₁ i) → M₂) (h₁ h₂) : ⇑(⟨f, h₁, h₂⟩ : MultilinearMap R M₁ M₂) = f :=
  rfl


theorem congr_fun {f g : MultilinearMap R M₁ M₂} (h : f = g) (x : ∀ i, M₁ i) : f x = g x :=
  DFunLike.congr_fun h x


nonrec theorem congr_arg (f : MultilinearMap R M₁ M₂) {x y : ∀ i, M₁ i} (h : x = y) : f x = f y :=
  DFunLike.congr_arg f h


theorem coe_injective : Injective ((↑) : MultilinearMap R M₁ M₂ → (∀ i, M₁ i) → M₂) :=
  DFunLike.coe_injective


@[norm_cast]
theorem coe_inj {f g : MultilinearMap R M₁ M₂} : (f : (∀ i, M₁ i) → M₂) = g ↔ f = g :=
  DFunLike.coe_fn_eq


@[ext]
theorem ext {f f' : MultilinearMap R M₁ M₂} (H : ∀ x, f x = f' x) : f = f' :=
  DFunLike.ext _ _ H


@[simp]
theorem mk_coe (f : MultilinearMap R M₁ M₂) (h₁ h₂) :
    (⟨f, h₁, h₂⟩ : MultilinearMap R M₁ M₂) = f := rfl


@[simp]
protected theorem map_update_add [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (x y : M₁ i) :
    f (update m i (x + y)) = f (update m i x) + f (update m i y) :=
  f.map_update_add' m i x y


@[deprecated (since := "2024-11-03")] protected alias map_add := MultilinearMap.map_update_add

@[deprecated (since := "2024-11-03")] protected alias map_add' := MultilinearMap.map_update_add


/-- Earlier, this name was used by what is now called `MultilinearMap.map_update_smul_left`. -/
@[simp]
protected theorem map_update_smul [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (c : R) (x : M₁ i) :
    f (update m i (c • x)) = c • f (update m i x) :=
  f.map_update_smul' m i c x


@[deprecated (since := "2024-11-03")] protected alias map_smul := MultilinearMap.map_update_smul

@[deprecated (since := "2024-11-03")] protected alias map_smul' := MultilinearMap.map_update_smul


theorem map_coord_zero {m : ∀ i, M₁ i} (i : ι) (h : m i = 0) : f m = 0 := by
  classical
    have : (0 : R) • (0 : M₁ i) = 0 := by simp
    rw [← update_eq_self i m, h, ← this, f.map_update_smul, zero_smul R (M := M₂)]


@[simp]
theorem map_update_zero [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) : f (update m i 0) = 0 :=
  f.map_coord_zero i (update_self i 0 m)


@[simp]
theorem map_zero [Nonempty ι] : f 0 = 0 := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : Nonempty ι
    ⊢ Eq (f 0) 0
  -/
  obtain ⟨i, _⟩ : ∃ i : ι, i ∈ Set.univ := Set.exists_mem_of_nonempty ι
  /-
    case intro
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : Nonempty ι
    i : ι
    h✝ : Membership.mem Set.univ i
    ⊢ Eq (f 0) 0
  -/
  exact map_coord_zero f i rfl
  /-
    🎉 no goals
  -/


instance : Add (MultilinearMap R M₁ M₂) :=
  ⟨fun f f' =>
                                            /-
                                              R : Type uR
                                              S : Type uS
                                              ι : Type uι
                                              n : Nat
                                              M : Fin n.succ → Type v
                                              M₁ : ι → Type v₁
                                              M₂ : Type v₂
                                              M₃ : Type v₃
                                              M' : Type v'
                                              inst✝¹¹ : Semiring R
                                              inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                                              inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                                              inst✝⁸ : AddCommMonoid M₂
                                              inst✝⁷ : AddCommMonoid M₃
                                              inst✝⁶ : AddCommMonoid M'
                                              inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                                              inst✝⁴ : (i : ι) → Module R (M₁ i)
                                              inst✝³ : Module R M₂
                                              inst✝² : Module R M₃
                                              inst✝¹ : Module R M'
                                              f✝ f'✝ f f' : MultilinearMap R M₁ M₂
                                              inst✝ : DecidableEq ι
                                              m : (i : ι) → M₁ i
                                              i : ι
                                              x y : M₁ i
                                              ⊢ Eq ((fun x => HAdd.hAdd (f x) (f' x)) (Function.update m i (HAdd.hAdd x y))) …
                                            -/
    ⟨fun x => f x + f' x, fun m i x y => by simp [add_left_comm, add_assoc], fun m i c x => by
                                            /-
                                              🎉 no goals
                                            -/
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁸ : AddCommMonoid M₂
        inst✝⁷ : AddCommMonoid M₃
        inst✝⁶ : AddCommMonoid M'
        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
        inst✝⁴ : (i : ι) → Module R (M₁ i)
        inst✝³ : Module R M₂
        inst✝² : Module R M₃
        inst✝¹ : Module R M'
        f✝ f'✝ f f' : MultilinearMap R M₁ M₂
        inst✝ : DecidableEq ι
        m : (i : ι) → M₁ i
        i : ι
        c : R
        x : M₁ i
        ⊢ Eq ((fun x => HAdd.hAdd (f x) (f' x)) (Function.update m i (HSMul.hSMul c x) …
      -/
      simp [smul_add]⟩⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem add_apply (m : ∀ i, M₁ i) : (f + f') m = f m + f' m :=
  rfl


instance : Zero (MultilinearMap R M₁ M₂) :=
                                  /-
                                    R : Type uR
                                    S : Type uS
                                    ι : Type uι
                                    n : Nat
                                    M : Fin n.succ → Type v
                                    M₁ : ι → Type v₁
                                    M₂ : Type v₂
                                    M₃ : Type v₃
                                    M' : Type v'
                                    inst✝¹¹ : Semiring R
                                    inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                                    inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                                    inst✝⁸ : AddCommMonoid M₂
                                    inst✝⁷ : AddCommMonoid M₃
                                    inst✝⁶ : AddCommMonoid M'
                                    inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                                    inst✝⁴ : (i : ι) → Module R (M₁ i)
                                    inst✝³ : Module R M₂
                                    inst✝² : Module R M₃
                                    inst✝¹ : Module R M'
                                    f f' : MultilinearMap R M₁ M₂
                                    inst✝ : DecidableEq ι
                                    x✝³ : (i : ι) → M₁ i
                                    x✝² : ι
                                    x✝¹ x✝ : M₁ x✝²
                                    ⊢ Eq ((fun x => 0) (Function.update x✝³ x✝² (HAdd.hAdd x✝¹ x✝))) (HAdd.hAdd (( …
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  ⟨⟨fun _ => 0, fun _ _ _ _ => by simp, fun _ _ c _ => by simp⟩⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


instance : Inhabited (MultilinearMap R M₁ M₂) :=
  ⟨0⟩


@[simp]
theorem zero_apply (m : ∀ i, M₁ i) : (0 : MultilinearMap R M₁ M₂) m = 0 :=
  rfl


instance : SMul R' (MultilinearMap A M₁ M₂) :=
  ⟨fun c f =>
                                         /-
                                           R : Type uR
                                           S : Type uS
                                           ι : Type uι
                                           n : Nat
                                           M : Fin n.succ → Type v
                                           M₁ : ι → Type v₁
                                           M₂ : Type v₂
                                           M₃ : Type v₃
                                           M' : Type v'
                                           inst✝¹⁷ : Semiring R
                                           inst✝¹⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
                                           inst✝¹⁵ : (i : ι) → AddCommMonoid (M₁ i)
                                           inst✝¹⁴ : AddCommMonoid M₂
                                           inst✝¹³ : AddCommMonoid M₃
                                           inst✝¹² : AddCommMonoid M'
                                           inst✝¹¹ : (i : Fin n.succ) → Module R (M i)
                                           inst✝¹⁰ : (i : ι) → Module R (M₁ i)
                                           inst✝⁹ : Module R M₂
                                           inst✝⁸ : Module R M₃
                                           inst✝⁷ : Module R M'
                                           f✝ f' : MultilinearMap R M₁ M₂
                                           R' : Type u_1
                                           A : Type u_2
                                           inst✝⁶ : Monoid R'
                                           inst✝⁵ : Semiring A
                                           inst✝⁴ : (i : ι) → Module A (M₁ i)
                                           inst✝³ : DistribMulAction R' M₂
                                           inst✝² : Module A M₂
                                           inst✝¹ : SMulCommClass A R' M₂
                                           c : R'
                                           f : MultilinearMap A M₁ M₂
                                           inst✝ : DecidableEq ι
                                           m : (i : ι) → M₁ i
                                           i : ι
                                           x y : M₁ i
                                           ⊢ Eq ((fun m => HSMul.hSMul c (f m)) (Function.update m i (HAdd.hAdd x y))) (H …
                                         -/
    ⟨fun m => c • f m, fun m i x y => by simp [smul_add], fun l i x d => by
                                         /-
                                           🎉 no goals
                                         -/
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝¹⁷ : Semiring R
        inst✝¹⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝¹⁵ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝¹⁴ : AddCommMonoid M₂
        inst✝¹³ : AddCommMonoid M₃
        inst✝¹² : AddCommMonoid M'
        inst✝¹¹ : (i : Fin n.succ) → Module R (M i)
        inst✝¹⁰ : (i : ι) → Module R (M₁ i)
        inst✝⁹ : Module R M₂
        inst✝⁸ : Module R M₃
        inst✝⁷ : Module R M'
        f✝ f' : MultilinearMap R M₁ M₂
        R' : Type u_1
        A : Type u_2
        inst✝⁶ : Monoid R'
        inst✝⁵ : Semiring A
        inst✝⁴ : (i : ι) → Module A (M₁ i)
        inst✝³ : DistribMulAction R' M₂
        inst✝² : Module A M₂
        inst✝¹ : SMulCommClass A R' M₂
        c : R'
        f : MultilinearMap A M₁ M₂
        inst✝ : DecidableEq ι
        l : (i : ι) → M₁ i
        i : ι
        x : A
        d : M₁ i
        ⊢ Eq ((fun m => HSMul.hSMul c (f m)) (Function.update l i (HSMul.hSMul x d)))  …
      -/
      simp [← smul_comm x c (_ : M₂)]⟩⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem smul_apply (f : MultilinearMap A M₁ M₂) (c : R') (m : ∀ i, M₁ i) : (c • f) m = c • f m :=
  rfl


theorem coe_smul (c : R') (f : MultilinearMap A M₁ M₂) : ⇑(c • f) = c • (⇑ f) :=
  rfl


instance addCommMonoid : AddCommMonoid (MultilinearMap R M₁ M₂) :=
  coe_injective.addCommMonoid _ rfl (fun _ _ => rfl) fun _ _ => rfl


/-- Coercion of a multilinear map to a function as an additive monoid homomorphism. -/
@[simps] def coeAddMonoidHom : MultilinearMap R M₁ M₂ →+ (((i : ι) → M₁ i) → M₂) where
  toFun := DFunLike.coe; map_zero' := rfl; map_add' _ _ := rfl


@[simp]
theorem coe_sum {α : Type*} (f : α → MultilinearMap R M₁ M₂) (s : Finset α) :
    ⇑(∑ a ∈ s, f a) = ∑ a ∈ s, ⇑(f a) :=
  map_sum coeAddMonoidHom f s


theorem sum_apply {α : Type*} (f : α → MultilinearMap R M₁ M₂) (m : ∀ i, M₁ i) {s : Finset α} :
                                            /-
                                              R : Type uR
                                              ι : Type uι
                                              M₁ : ι → Type v₁
                                              M₂ : Type v₂
                                              inst✝⁴ : Semiring R
                                              inst✝³ : (i : ι) → AddCommMonoid (M₁ i)
                                              inst✝² : AddCommMonoid M₂
                                              inst✝¹ : (i : ι) → Module R (M₁ i)
                                              inst✝ : Module R M₂
                                              α : Type u_1
                                              f : α → MultilinearMap R M₁ M₂
                                              m : (i : ι) → M₁ i
                                              s : Finset α
                                              ⊢ Eq ((s.sum fun a => f a) m) (s.sum fun a => (f a) m)
                                            -/
    (∑ a ∈ s, f a) m = ∑ a ∈ s, f a m := by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- If `f` is a multilinear map, then `f.toLinearMap m i` is the linear map obtained by fixing all
coordinates but `i` equal to those of `m`, and varying the `i`-th coordinate. -/
@[simps]
def toLinearMap [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) : M₁ i →ₗ[R] M₂ where
  toFun x := f (update m i x)
                     /-
                       R : Type uR
                       S : Type uS
                       ι : Type uι
                       n : Nat
                       M : Fin n.succ → Type v
                       M₁ : ι → Type v₁
                       M₂ : Type v₂
                       M₃ : Type v₃
                       M' : Type v'
                       inst✝¹¹ : Semiring R
                       inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                       inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                       inst✝⁸ : AddCommMonoid M₂
                       inst✝⁷ : AddCommMonoid M₃
                       inst✝⁶ : AddCommMonoid M'
                       inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                       inst✝⁴ : (i : ι) → Module R (M₁ i)
                       inst✝³ : Module R M₂
                       inst✝² : Module R M₃
                       inst✝¹ : Module R M'
                       f f' : MultilinearMap R M₁ M₂
                       inst✝ : DecidableEq ι
                       m : (i : ι) → M₁ i
                       i : ι
                       x y : M₁ i
                       ⊢ Eq ((fun x => f (Function.update m i x)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x …
                     -/
  map_add' x y := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type uR
                        S : Type uS
                        ι : Type uι
                        n : Nat
                        M : Fin n.succ → Type v
                        M₁ : ι → Type v₁
                        M₂ : Type v₂
                        M₃ : Type v₃
                        M' : Type v'
                        inst✝¹¹ : Semiring R
                        inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                        inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                        inst✝⁸ : AddCommMonoid M₂
                        inst✝⁷ : AddCommMonoid M₃
                        inst✝⁶ : AddCommMonoid M'
                        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                        inst✝⁴ : (i : ι) → Module R (M₁ i)
                        inst✝³ : Module R M₂
                        inst✝² : Module R M₃
                        inst✝¹ : Module R M'
                        f f' : MultilinearMap R M₁ M₂
                        inst✝ : DecidableEq ι
                        m : (i : ι) → M₁ i
                        i : ι
                        c : R
                        x : M₁ i
                        ⊢ Eq ({ toFun := fun x => f (Function.update m i x), map_add' := ⋯ }.toFun (HS …
                      -/
  map_smul' c x := by simp
                      /-
                        🎉 no goals
                      -/


/-- The cartesian product of two multilinear maps, as a multilinear map. -/
@[simps]
def prod (f : MultilinearMap R M₁ M₂) (g : MultilinearMap R M₁ M₃) :
    MultilinearMap R M₁ (M₂ × M₃) where
  toFun m := (f m, g m)
                                /-
                                  R : Type uR
                                  S : Type uS
                                  ι : Type uι
                                  n : Nat
                                  M : Fin n.succ → Type v
                                  M₁ : ι → Type v₁
                                  M₂ : Type v₂
                                  M₃ : Type v₃
                                  M' : Type v'
                                  inst✝¹¹ : Semiring R
                                  inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                                  inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                                  inst✝⁸ : AddCommMonoid M₂
                                  inst✝⁷ : AddCommMonoid M₃
                                  inst✝⁶ : AddCommMonoid M'
                                  inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                                  inst✝⁴ : (i : ι) → Module R (M₁ i)
                                  inst✝³ : Module R M₂
                                  inst✝² : Module R M₃
                                  inst✝¹ : Module R M'
                                  f✝ f' f : MultilinearMap R M₁ M₂
                                  g : MultilinearMap R M₁ M₃
                                  inst✝ : DecidableEq ι
                                  m : (i : ι) → M₁ i
                                  i : ι
                                  x y : M₁ i
                                  ⊢ Eq ((fun m => { fst := f m, snd := g m }) (Function.update m i (HAdd.hAdd x  …
                                -/
  map_update_add' m i x y := by simp
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   R : Type uR
                                   S : Type uS
                                   ι : Type uι
                                   n : Nat
                                   M : Fin n.succ → Type v
                                   M₁ : ι → Type v₁
                                   M₂ : Type v₂
                                   M₃ : Type v₃
                                   M' : Type v'
                                   inst✝¹¹ : Semiring R
                                   inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                                   inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                                   inst✝⁸ : AddCommMonoid M₂
                                   inst✝⁷ : AddCommMonoid M₃
                                   inst✝⁶ : AddCommMonoid M'
                                   inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                                   inst✝⁴ : (i : ι) → Module R (M₁ i)
                                   inst✝³ : Module R M₂
                                   inst✝² : Module R M₃
                                   inst✝¹ : Module R M'
                                   f✝ f' f : MultilinearMap R M₁ M₂
                                   g : MultilinearMap R M₁ M₃
                                   inst✝ : DecidableEq ι
                                   m : (i : ι) → M₁ i
                                   i : ι
                                   c : R
                                   x : M₁ i
                                   ⊢ Eq ((fun m => { fst := f m, snd := g m }) (Function.update m i (HSMul.hSMul  …
                                 -/
  map_update_smul' m i c x := by simp
                                 /-
                                   🎉 no goals
                                 -/


/-- Combine a family of multilinear maps with the same domain and codomains `M' i` into a
multilinear map taking values in the space of functions `∀ i, M' i`. -/
@[simps]
def pi {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)] [∀ i, Module R (M' i)]
    (f : ∀ i, MultilinearMap R M₁ (M' i)) : MultilinearMap R M₁ (∀ i, M' i) where
  toFun m i := f i m
  map_update_add' _ _ _ _ := funext fun j => (f j).map_update_add _ _ _ _
  map_update_smul' _ _ _ _ := funext fun j => (f j).map_update_smul _ _ _ _


/-- Equivalence between linear maps `M₂ →ₗ[R] M₃` and one-multilinear maps. -/
@[simps]
def ofSubsingleton [Subsingleton ι] (i : ι) :
    (M₂ →ₗ[R] M₃) ≃ MultilinearMap R (fun _ : ι ↦ M₂) M₃ where
  toFun f :=
    { toFun := fun x ↦ f (x i)
                            /-
                              R : Type uR
                              S : Type uS
                              ι : Type uι
                              n : Nat
                              M : Fin n.succ → Type v
                              M₁ : ι → Type v₁
                              M₂ : Type v₂
                              M₃ : Type v₃
                              M' : Type v'
                              inst✝¹¹ : Semiring R
                              inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                              inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                              inst✝⁸ : AddCommMonoid M₂
                              inst✝⁷ : AddCommMonoid M₃
                              inst✝⁶ : AddCommMonoid M'
                              inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                              inst✝⁴ : (i : ι) → Module R (M₁ i)
                              inst✝³ : Module R M₂
                              inst✝² : Module R M₃
                              inst✝¹ : Module R M'
                              f✝ f' : MultilinearMap R M₁ M₂
                              inst✝ : Subsingleton ι
                              i : ι
                              f : LinearMap (RingHom.id R) M₂ M₃
                              ⊢ ∀ [inst : DecidableEq ι] (m : ι → M₂) (i_1 : ι) (x y : M₂), Eq ((fun x => f  …
                            -/
      map_update_add' := by intros; simp [update_eq_const_of_subsingleton]
                                    /-
                                      🎉 no goals
                                    -/
                             /-
                               R : Type uR
                               S : Type uS
                               ι : Type uι
                               n : Nat
                               M : Fin n.succ → Type v
                               M₁ : ι → Type v₁
                               M₂ : Type v₂
                               M₃ : Type v₃
                               M' : Type v'
                               inst✝¹¹ : Semiring R
                               inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                               inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                               inst✝⁸ : AddCommMonoid M₂
                               inst✝⁷ : AddCommMonoid M₃
                               inst✝⁶ : AddCommMonoid M'
                               inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                               inst✝⁴ : (i : ι) → Module R (M₁ i)
                               inst✝³ : Module R M₂
                               inst✝² : Module R M₃
                               inst✝¹ : Module R M'
                               f✝ f' : MultilinearMap R M₁ M₂
                               inst✝ : Subsingleton ι
                               i : ι
                               f : LinearMap (RingHom.id R) M₂ M₃
                               ⊢ ∀ [inst : DecidableEq ι] (m : ι → M₂) (i_1 : ι) (c : R) (x : M₂), Eq ((fun x …
                             -/
      map_update_smul' := by intros; simp [update_eq_const_of_subsingleton] }
                                     /-
                                       🎉 no goals
                                     -/
  invFun f :=
    { toFun := fun x ↦ f fun _ ↦ x
      map_add' := fun x y ↦ by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹¹ : Semiring R
          inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝⁸ : AddCommMonoid M₂
          inst✝⁷ : AddCommMonoid M₃
          inst✝⁶ : AddCommMonoid M'
          inst✝⁵ : (i : Fin n.succ) → Module R (M i)
          inst✝⁴ : (i : ι) → Module R (M₁ i)
          inst✝³ : Module R M₂
          inst✝² : Module R M₃
          inst✝¹ : Module R M'
          f✝ f' : MultilinearMap R M₁ M₂
          inst✝ : Subsingleton ι
          i : ι
          f : MultilinearMap R (fun x => M₂) M₃
          x y : M₂
          ⊢ Eq ((fun x => f fun x_1 => x) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f fun x …
        -/
        simpa [update_eq_const_of_subsingleton] using f.map_update_add 0 i x y
        /-
          🎉 no goals
        -/
      map_smul' := fun c x ↦ by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹¹ : Semiring R
          inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝⁸ : AddCommMonoid M₂
          inst✝⁷ : AddCommMonoid M₃
          inst✝⁶ : AddCommMonoid M'
          inst✝⁵ : (i : Fin n.succ) → Module R (M i)
          inst✝⁴ : (i : ι) → Module R (M₁ i)
          inst✝³ : Module R M₂
          inst✝² : Module R M₃
          inst✝¹ : Module R M'
          f✝ f' : MultilinearMap R M₁ M₂
          inst✝ : Subsingleton ι
          i : ι
          f : MultilinearMap R (fun x => M₂) M₃
          c : R
          x : M₂
          ⊢ Eq ({ toFun := fun x => f fun x_1 => x, map_add' := ⋯ }.toFun (HSMul.hSMul c …
        -/
        simpa [update_eq_const_of_subsingleton] using f.map_update_smul 0 i c x }
        /-
          🎉 no goals
        -/
  left_inv _ := rfl
                    /-
                      R : Type uR
                      S : Type uS
                      ι : Type uι
                      n : Nat
                      M : Fin n.succ → Type v
                      M₁ : ι → Type v₁
                      M₂ : Type v₂
                      M₃ : Type v₃
                      M' : Type v'
                      inst✝¹¹ : Semiring R
                      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
                      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                      inst✝⁸ : AddCommMonoid M₂
                      inst✝⁷ : AddCommMonoid M₃
                      inst✝⁶ : AddCommMonoid M'
                      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                      inst✝⁴ : (i : ι) → Module R (M₁ i)
                      inst✝³ : Module R M₂
                      inst✝² : Module R M₃
                      inst✝¹ : Module R M'
                      f✝ f' : MultilinearMap R M₁ M₂
                      inst✝ : Subsingleton ι
                      i : ι
                      f : MultilinearMap R (fun x => M₂) M₃
                      ⊢ Eq ((fun f => { toFun := fun x => f (x i), map_update_add' := ⋯, map_update_ …
                    -/
  right_inv f := by ext x; refine congr_arg f ?_; exact (eq_const_of_subsingleton _ _).symm
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The constant map is multilinear when `ι` is empty. -/
-- Porting note: Removed [simps] & added simpNF-approved version of the generated lemma manually.
@[simps (config := .asFn)]
def constOfIsEmpty [IsEmpty ι] (m : M₂) : MultilinearMap R M₁ M₂ where
  toFun := Function.const _ m
  map_update_add' _ := isEmptyElim
  map_update_smul' _ := isEmptyElim


/-- Given a multilinear map `f` on `n` variables (parameterized by `Fin n`) and a subset `s` of `k`
of these variables, one gets a new multilinear map on `Fin k` by varying these variables, and fixing
the other ones equal to a given value `z`. It is denoted by `f.restr s hk z`, where `hk` is a
proof that the cardinality of `s` is `k`. The implicit identification between `Fin k` and `s` that
we use is the canonical (increasing) bijection. -/
def restr {k n : ℕ} (f : MultilinearMap R (fun _ : Fin n => M') M₂) (s : Finset (Fin n))
    (hk : #s = k) (z : M') : MultilinearMap R (fun _ : Fin k => M') M₂ where
  toFun v := f fun j => if h : j ∈ s then v ((s.orderIsoOfFin hk).symm ⟨j, h⟩) else z
  /- Porting note: The proofs of the following two lemmas used to only use `erw` followed by `simp`,
  but it seems `erw` no longer unfolds or unifies well enough to work without more help. -/
  map_update_add' v i x y := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n✝ : Nat
      M : Fin n✝.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n✝.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n✝.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      k n : Nat
      f : MultilinearMap R (fun x => M') M₂
      s : Finset (Fin n)
      hk : Eq s.card k
      z : M'
      inst✝ : DecidableEq (Fin k)
      v : Fin k → M'
      i : Fin k
      x y : M'
      ⊢ Eq ((fun v => f fun j => dite (Membership.mem s j) (fun h => v ((s.orderIsoO …
    -/
    have : DFunLike.coe (s.orderIsoOfFin hk).symm = (s.orderIsoOfFin hk).toEquiv.symm := rfl
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n✝ : Nat
      M : Fin n✝.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n✝.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n✝.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      k n : Nat
      f : MultilinearMap R (fun x => M') M₂
      s : Finset (Fin n)
      hk : Eq s.card k
      z : M'
      inst✝ : DecidableEq (Fin k)
      v : Fin k → M'
      i : Fin k
      x y : M'
      this : Eq ⇑(s.orderIsoOfFin hk).symm ⇑(s.orderIsoOfFin hk).symm
      ⊢ Eq ((fun v => f fun j => dite (Membership.mem s j) (fun h => v ((s.orderIsoO …
    -/
    simp only [this]
    erw [dite_comp_equiv_update (s.orderIsoOfFin hk).toEquiv,
      dite_comp_equiv_update (s.orderIsoOfFin hk).toEquiv,
      dite_comp_equiv_update (s.orderIsoOfFin hk).toEquiv]
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n✝ : Nat
      M : Fin n✝.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n✝.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n✝.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      k n : Nat
      f : MultilinearMap R (fun x => M') M₂
      s : Finset (Fin n)
      hk : Eq s.card k
      z : M'
      inst✝ : DecidableEq (Fin k)
      v : Fin k → M'
      i : Fin k
      x y : M'
      this : Eq ⇑(s.orderIsoOfFin hk).symm ⇑(s.orderIsoOfFin hk).symm
      ⊢ Eq (f (Function.update (fun i => dite (Membership.mem s i) (fun h => v ((s.o …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_update_smul' v i c x := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n✝ : Nat
      M : Fin n✝.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n✝.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n✝.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      k n : Nat
      f : MultilinearMap R (fun x => M') M₂
      s : Finset (Fin n)
      hk : Eq s.card k
      z : M'
      inst✝ : DecidableEq (Fin k)
      v : Fin k → M'
      i : Fin k
      c : R
      x : M'
      ⊢ Eq ((fun v => f fun j => dite (Membership.mem s j) (fun h => v ((s.orderIsoO …
    -/
    have : DFunLike.coe (s.orderIsoOfFin hk).symm = (s.orderIsoOfFin hk).toEquiv.symm := rfl
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n✝ : Nat
      M : Fin n✝.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n✝.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n✝.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      k n : Nat
      f : MultilinearMap R (fun x => M') M₂
      s : Finset (Fin n)
      hk : Eq s.card k
      z : M'
      inst✝ : DecidableEq (Fin k)
      v : Fin k → M'
      i : Fin k
      c : R
      x : M'
      this : Eq ⇑(s.orderIsoOfFin hk).symm ⇑(s.orderIsoOfFin hk).symm
      ⊢ Eq ((fun v => f fun j => dite (Membership.mem s j) (fun h => v ((s.orderIsoO …
    -/
    simp only [this]
    erw [dite_comp_equiv_update (s.orderIsoOfFin hk).toEquiv,
      dite_comp_equiv_update (s.orderIsoOfFin hk).toEquiv]
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n✝ : Nat
      M : Fin n✝.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n✝.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n✝.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      k n : Nat
      f : MultilinearMap R (fun x => M') M₂
      s : Finset (Fin n)
      hk : Eq s.card k
      z : M'
      inst✝ : DecidableEq (Fin k)
      v : Fin k → M'
      i : Fin k
      c : R
      x : M'
      this : Eq ⇑(s.orderIsoOfFin hk).symm ⇑(s.orderIsoOfFin hk).symm
      ⊢ Eq (f (Function.update (fun i => dite (Membership.mem s i) (fun h => v ((s.o …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- In the specific case of multilinear maps on spaces indexed by `Fin (n+1)`, where one can build
an element of `∀ (i : Fin (n+1)), M i` using `cons`, one can express directly the additivity of a
multilinear map along the first variable. -/
theorem cons_add (f : MultilinearMap R M M₂) (m : ∀ i : Fin n, M i.succ) (x y : M 0) :
    f (cons (x + y) m) = f (cons x m) + f (cons y m) := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : Semiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    m : (i : Fin n) → M i.succ
    x y : M 0
    ⊢ Eq (f (Fin.cons (HAdd.hAdd x y) m)) (HAdd.hAdd (f (Fin.cons x m)) (f (Fin.co …
  -/
  simp_rw [← update_cons_zero x m (x + y), f.map_update_add, update_cons_zero]
  /-
    🎉 no goals
  -/


/-- In the specific case of multilinear maps on spaces indexed by `Fin (n+1)`, where one can build
an element of `∀ (i : Fin (n+1)), M i` using `cons`, one can express directly the multiplicativity
of a multilinear map along the first variable. -/
theorem cons_smul (f : MultilinearMap R M M₂) (m : ∀ i : Fin n, M i.succ) (c : R) (x : M 0) :
    f (cons (c • x) m) = c • f (cons x m) := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : Semiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    m : (i : Fin n) → M i.succ
    c : R
    x : M 0
    ⊢ Eq (f (Fin.cons (HSMul.hSMul c x) m)) (HSMul.hSMul c (f (Fin.cons x m)))
  -/
  simp_rw [← update_cons_zero x m (c • x), f.map_update_smul, update_cons_zero]
  /-
    🎉 no goals
  -/


/-- In the specific case of multilinear maps on spaces indexed by `Fin (n+1)`, where one can build
an element of `∀ (i : Fin (n+1)), M i` using `snoc`, one can express directly the additivity of a
multilinear map along the first variable. -/
theorem snoc_add (f : MultilinearMap R M M₂)
    (m : ∀ i : Fin n, M (castSucc i)) (x y : M (last n)) :
    f (snoc m (x + y)) = f (snoc m x) + f (snoc m y) := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : Semiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    m : (i : Fin n) → M i.castSucc
    x y : M (Fin.last n)
    ⊢ Eq (f (Fin.snoc m (HAdd.hAdd x y))) (HAdd.hAdd (f (Fin.snoc m x)) (f (Fin.sn …
  -/
  simp_rw [← update_snoc_last x m (x + y), f.map_update_add, update_snoc_last]
  /-
    🎉 no goals
  -/


/-- In the specific case of multilinear maps on spaces indexed by `Fin (n+1)`, where one can build
an element of `∀ (i : Fin (n+1)), M i` using `cons`, one can express directly the multiplicativity
of a multilinear map along the first variable. -/
theorem snoc_smul (f : MultilinearMap R M M₂) (m : ∀ i : Fin n, M (castSucc i)) (c : R)
    (x : M (last n)) : f (snoc m (c • x)) = c • f (snoc m x) := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : Semiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    m : (i : Fin n) → M i.castSucc
    c : R
    x : M (Fin.last n)
    ⊢ Eq (f (Fin.snoc m (HSMul.hSMul c x))) (HSMul.hSMul c (f (Fin.snoc m x)))
  -/
  simp_rw [← update_snoc_last x m (c • x), f.map_update_smul, update_snoc_last]
  /-
    🎉 no goals
  -/


/-- If `g` is a multilinear map and `f` is a collection of linear maps,
then `g (f₁ m₁, ..., fₙ mₙ)` is again a multilinear map, that we call
`g.compLinearMap f`. -/
def compLinearMap (g : MultilinearMap R M₁' M₂) (f : ∀ i, M₁ i →ₗ[R] M₁' i) :
    MultilinearMap R M₁ M₂ where
  toFun m := g fun i => f i (m i)
  map_update_add' m i x y := by
    have : ∀ j z, f j (update m i z j) = update (fun k => f k (m k)) i (f i z) j := fun j z =>
      Function.apply_update (fun k => f k) _ _ _ _
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝¹³ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M'
      inst✝⁹ : (i : Fin n.succ) → Module R (M i)
      inst✝⁸ : (i : ι) → Module R (M₁ i)
      inst✝⁷ : Module R M₂
      inst✝⁶ : Module R M₃
      inst✝⁵ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝⁴ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝³ : (i : ι) → Module R (M₁' i)
      M₁'' : ι → Type u_2
      inst✝² : (i : ι) → AddCommMonoid (M₁'' i)
      inst✝¹ : (i : ι) → Module R (M₁'' i)
      g : MultilinearMap R M₁' M₂
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      inst✝ : DecidableEq ι
      m : (i : ι) → M₁ i
      i : ι
      x y : M₁ i
      this : ∀ (j : ι) (z : M₁ i), Eq ((f j) (Function.update m i z j)) (Function.up …
      ⊢ Eq ((fun m => g fun i => (f i) (m i)) (Function.update m i (HAdd.hAdd x y))) …
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  map_update_smul' m i c x := by
    have : ∀ j z, f j (update m i z j) = update (fun k => f k (m k)) i (f i z) j := fun j z =>
      Function.apply_update (fun k => f k) _ _ _ _
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁵ : Semiring R
      inst✝¹⁴ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝¹³ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : AddCommMonoid M₃
      inst✝¹⁰ : AddCommMonoid M'
      inst✝⁹ : (i : Fin n.succ) → Module R (M i)
      inst✝⁸ : (i : ι) → Module R (M₁ i)
      inst✝⁷ : Module R M₂
      inst✝⁶ : Module R M₃
      inst✝⁵ : Module R M'
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝⁴ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝³ : (i : ι) → Module R (M₁' i)
      M₁'' : ι → Type u_2
      inst✝² : (i : ι) → AddCommMonoid (M₁'' i)
      inst✝¹ : (i : ι) → Module R (M₁'' i)
      g : MultilinearMap R M₁' M₂
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      inst✝ : DecidableEq ι
      m : (i : ι) → M₁ i
      i : ι
      c : R
      x : M₁ i
      this : ∀ (j : ι) (z : M₁ i), Eq ((f j) (Function.update m i z j)) (Function.up …
      ⊢ Eq ((fun m => g fun i => (f i) (m i)) (Function.update m i (HSMul.hSMul c x) …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


@[simp]
theorem compLinearMap_apply (g : MultilinearMap R M₁' M₂) (f : ∀ i, M₁ i →ₗ[R] M₁' i)
    (m : ∀ i, M₁ i) : g.compLinearMap f m = g fun i => f i (m i) :=
  rfl


/-- Composing a multilinear map twice with a linear map in each argument is
the same as composing with their composition. -/
theorem compLinearMap_assoc (g : MultilinearMap R M₁'' M₂) (f₁ : ∀ i, M₁' i →ₗ[R] M₁'' i)
    (f₂ : ∀ i, M₁ i →ₗ[R] M₁' i) :
    (g.compLinearMap f₁).compLinearMap f₂ = g.compLinearMap fun i => f₁ i ∘ₗ f₂ i :=
  rfl


/-- Composing the zero multilinear map with a linear map in each argument. -/
@[simp]
theorem zero_compLinearMap (f : ∀ i, M₁ i →ₗ[R] M₁' i) :
    (0 : MultilinearMap R M₁' M₂).compLinearMap f = 0 :=
  ext fun _ => rfl


/-- Composing a multilinear map with the identity linear map in each argument. -/
@[simp]
theorem compLinearMap_id (g : MultilinearMap R M₁' M₂) :
    (g.compLinearMap fun _ => LinearMap.id) = g :=
  ext fun _ => rfl


/-- Composing with a family of surjective linear maps is injective. -/
theorem compLinearMap_injective (f : ∀ i, M₁ i →ₗ[R] M₁' i) (hf : ∀ i, Surjective (f i)) :
    Injective fun g : MultilinearMap R M₁' M₂ => g.compLinearMap f := fun g₁ g₂ h =>
  ext fun x => by
    simpa [fun i => surjInv_eq (hf i)]
      using MultilinearMap.ext_iff.mp h fun i => surjInv (hf i) (x i)


theorem compLinearMap_inj (f : ∀ i, M₁ i →ₗ[R] M₁' i) (hf : ∀ i, Surjective (f i))
    (g₁ g₂ : MultilinearMap R M₁' M₂) : g₁.compLinearMap f = g₂.compLinearMap f ↔ g₁ = g₂ :=
  (compLinearMap_injective _ hf).eq_iff


/-- Composing a multilinear map with a linear equiv on each argument gives the zero map
if and only if the multilinear map is the zero map. -/
@[simp]
theorem comp_linearEquiv_eq_zero_iff (g : MultilinearMap R M₁' M₂) (f : ∀ i, M₁ i ≃ₗ[R] M₁' i) :
    (g.compLinearMap fun i => (f i : M₁ i →ₗ[R] M₁' i)) = 0 ↔ g = 0 := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    M₁' : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (M₁' i)
    inst✝ : (i : ι) → Module R (M₁' i)
    g : MultilinearMap R M₁' M₂
    f : (i : ι) → LinearEquiv (RingHom.id R) (M₁ i) (M₁' i)
    ⊢ Iff (Eq (g.compLinearMap fun i => ↑(f i)) 0) (Eq g 0)
  -/
  set f' := fun i => (f i : M₁ i →ₗ[R] M₁' i)
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    M₁' : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (M₁' i)
    inst✝ : (i : ι) → Module R (M₁' i)
    g : MultilinearMap R M₁' M₂
    f : (i : ι) → LinearEquiv (RingHom.id R) (M₁ i) (M₁' i)
    f' : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) := fun i => ↑(f i)
    ⊢ Iff (Eq (g.compLinearMap f') 0) (Eq g 0)
  -/
  rw [← zero_compLinearMap f', compLinearMap_inj f' fun i => (f i).surjective]
  /-
    🎉 no goals
  -/


/-- If one adds to a vector `m'` another vector `m`, but only for coordinates in a finset `t`, then
the image under a multilinear map `f` is the sum of `f (s.piecewise m m')` along all subsets `s` of
`t`. This is mainly an auxiliary statement to prove the result when `t = univ`, given in
`map_add_univ`, although it can be useful in its own right as it does not require the index set `ι`
to be finite. -/
theorem map_piecewise_add [DecidableEq ι] (m m' : ∀ i, M₁ i) (t : Finset ι) :
    f (t.piecewise (m + m') m') = ∑ s ∈ t.powerset, f (s.piecewise m m') := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m m' : (i : ι) → M₁ i
    t : Finset ι
    ⊢ Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.powerset.sum fun s => f (s.piece …
  -/
  revert m'
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t : Finset ι
    ⊢ ∀ (m' : (i : ι) → M₁ i), Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.powerse …
  -/
  refine Finset.induction_on t (by simp) ?_
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t : Finset ι
    ⊢ ∀ ⦃a : ι⦄ {s : Finset ι}, Not (Membership.mem s a) → (∀ (m' : (i : ι) → M₁ i …
  -/
  intro i t hit Hrec m'
  have A : (insert i t).piecewise (m + m') m' = update (t.piecewise (m + m') m') i (m i + m' i) :=
    t.piecewise_insert _ _ _
  have B : update (t.piecewise (m + m') m') i (m' i) = t.piecewise (m + m') m' := by
    ext j
    by_cases h : j = i
    · rw [h]
      simp [hit]
    · simp [h]
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t✝ : Finset ι
    i : ι
    t : Finset ι
    hit : Not (Membership.mem t i)
    Hrec : ∀ (m' : (i : ι) → M₁ i), Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.po …
    m' : (i : ι) → M₁ i
    A : Eq ((Insert.insert i t).piecewise (HAdd.hAdd m m') m') (Function.update (t …
    B : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m' i)) (t.piecewi …
    ⊢ Eq (f ((Insert.insert i t).piecewise (HAdd.hAdd m m') m')) ((Insert.insert i …
  -/
  let m'' := update m' i (m i)
  have C : update (t.piecewise (m + m') m') i (m i) = t.piecewise (m + m'') m'' := by
    ext j
    by_cases h : j = i
    · rw [h]
      simp [m'', hit]
    · by_cases h' : j ∈ t <;> simp [m'', h, hit, h']
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t✝ : Finset ι
    i : ι
    t : Finset ι
    hit : Not (Membership.mem t i)
    Hrec : ∀ (m' : (i : ι) → M₁ i), Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.po …
    m' : (i : ι) → M₁ i
    A : Eq ((Insert.insert i t).piecewise (HAdd.hAdd m m') m') (Function.update (t …
    B : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m' i)) (t.piecewi …
    m'' : (a : ι) → M₁ a := Function.update m' i (m i)
    C : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m i)) (t.piecewis …
    ⊢ Eq (f ((Insert.insert i t).piecewise (HAdd.hAdd m m') m')) ((Insert.insert i …
  -/
  rw [A, f.map_update_add, B, C, Finset.sum_powerset_insert hit, Hrec, Hrec, add_comm (_ : M₂)]
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t✝ : Finset ι
    i : ι
    t : Finset ι
    hit : Not (Membership.mem t i)
    Hrec : ∀ (m' : (i : ι) → M₁ i), Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.po …
    m' : (i : ι) → M₁ i
    A : Eq ((Insert.insert i t).piecewise (HAdd.hAdd m m') m') (Function.update (t …
    B : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m' i)) (t.piecewi …
    m'' : (a : ι) → M₁ a := Function.update m' i (m i)
    C : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m i)) (t.piecewis …
    ⊢ Eq (HAdd.hAdd (t.powerset.sum fun s => f (s.piecewise m m')) (t.powerset.sum …
  -/
  congr 1
  /-
    case e_a
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t✝ : Finset ι
    i : ι
    t : Finset ι
    hit : Not (Membership.mem t i)
    Hrec : ∀ (m' : (i : ι) → M₁ i), Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.po …
    m' : (i : ι) → M₁ i
    A : Eq ((Insert.insert i t).piecewise (HAdd.hAdd m m') m') (Function.update (t …
    B : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m' i)) (t.piecewi …
    m'' : (a : ι) → M₁ a := Function.update m' i (m i)
    C : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m i)) (t.piecewis …
    ⊢ Eq (t.powerset.sum fun s => f (s.piecewise m m'')) (t.powerset.sum fun t =>  …
  -/
  refine Finset.sum_congr rfl fun s hs => ?_
  have : (insert i s).piecewise m m' = s.piecewise m m'' := by
    ext j
    by_cases h : j = i
    · rw [h]
      simp [m'', Finset.not_mem_of_mem_powerset_of_not_mem hs hit]
    · by_cases h' : j ∈ s <;> simp [m'', h, h']
  /-
    case e_a
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    t✝ : Finset ι
    i : ι
    t : Finset ι
    hit : Not (Membership.mem t i)
    Hrec : ∀ (m' : (i : ι) → M₁ i), Eq (f (t.piecewise (HAdd.hAdd m m') m')) (t.po …
    m' : (i : ι) → M₁ i
    A : Eq ((Insert.insert i t).piecewise (HAdd.hAdd m m') m') (Function.update (t …
    B : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m' i)) (t.piecewi …
    m'' : (a : ι) → M₁ a := Function.update m' i (m i)
    C : Eq (Function.update (t.piecewise (HAdd.hAdd m m') m') i (m i)) (t.piecewis …
    s : Finset ι
    hs : Membership.mem t.powerset s
    this : Eq ((Insert.insert i s).piecewise m m') (s.piecewise m m'')
    ⊢ Eq (f (s.piecewise m m'')) (f ((Insert.insert i s).piecewise m m'))
  -/
  rw [this]
  /-
    🎉 no goals
  -/


/-- Additivity of a multilinear map along all coordinates at the same time,
writing `f (m + m')` as the sum of `f (s.piecewise m m')` over all sets `s`. -/
theorem map_add_univ [DecidableEq ι] [Fintype ι] (m m' : ∀ i, M₁ i) :
    f (m + m') = ∑ s : Finset ι, f (s.piecewise m m') := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    m m' : (i : ι) → M₁ i
    ⊢ Eq (f (HAdd.hAdd m m')) (Finset.univ.sum fun s => f (s.piecewise m m'))
  -/
  simpa using f.map_piecewise_add m m' Finset.univ
  /-
    🎉 no goals
  -/


/-- If `f` is multilinear, then `f (Σ_{j₁ ∈ A₁} g₁ j₁, ..., Σ_{jₙ ∈ Aₙ} gₙ jₙ)` is the sum of
`f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions with `r 1 ∈ A₁`, ...,
`r n ∈ Aₙ`. This follows from multilinearity by expanding successively with respect to each
coordinate. Here, we give an auxiliary statement tailored for an inductive proof. Use instead
`map_sum_finset`. -/
theorem map_sum_finset_aux [DecidableEq ι] [Fintype ι] {n : ℕ} (h : (∑ i, #(A i)) = n) :
    (f fun i => ∑ j ∈ A i, g i j) = ∑ r ∈ piFinset A, f fun i => g i (r i) := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    A : (i : ι) → Finset (α i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    n : Nat
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  letI := fun i => Classical.decEq (α i)
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    A : (i : ι) → Finset (α i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    n : Nat
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  induction' n using Nat.strong_induction_on with n IH generalizing A
  -- If one of the sets is empty, then all the sums are zero
  /-
    case h
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  by_cases Ai_empty : ∃ i, A i = ∅
    /-
      case pos
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f : MultilinearMap R M₁ M₂
      α : ι → Type u_1
      g : (i : ι) → α i → M₁ i
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
      A : (i : ι) → Finset (α i)
      h : Eq (Finset.univ.sum fun i => (A i).card) n
      Ai_empty : Exists fun i => Eq (A i) EmptyCollection.emptyCollection
      ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
    -/
  · obtain ⟨i, hi⟩ : ∃ i, ∑ j ∈ A i, g i j = 0 := Ai_empty.imp fun i hi ↦ by simp [hi]
    /-
      case pos.intro
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f : MultilinearMap R M₁ M₂
      α : ι → Type u_1
      g : (i : ι) → α i → M₁ i
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
      A : (i : ι) → Finset (α i)
      h : Eq (Finset.univ.sum fun i => (A i).card) n
      Ai_empty : Exists fun i => Eq (A i) EmptyCollection.emptyCollection
      i : ι
      hi : Eq ((A i).sum fun j => g i j) 0
      ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
    -/
    have hpi : piFinset A = ∅ := by simpa
    /-
      case pos.intro
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁶ : Semiring R
      inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f : MultilinearMap R M₁ M₂
      α : ι → Type u_1
      g : (i : ι) → α i → M₁ i
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
      A : (i : ι) → Finset (α i)
      h : Eq (Finset.univ.sum fun i => (A i).card) n
      Ai_empty : Exists fun i => Eq (A i) EmptyCollection.emptyCollection
      i : ι
      hi : Eq ((A i).sum fun j => g i j) 0
      hpi : Eq (Fintype.piFinset A) EmptyCollection.emptyCollection
      ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
    -/
    rw [f.map_coord_zero i hi, hpi, Finset.sum_empty]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : Not (Exists fun i => Eq (A i) EmptyCollection.emptyCollection)
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  push_neg at Ai_empty
  -- Otherwise, if all sets are at most singletons, then they are exactly singletons and the result
  -- is again straightforward
  /-
    case neg
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  by_cases Ai_singleton : ∀ i, #(A i) ≤ 1
  · have Ai_card : ∀ i, #(A i) = 1 := by
      intro i
      have pos : #(A i) ≠ 0 := by simp [Finset.card_eq_zero, Ai_empty i]
      have : #(A i) ≤ 1 := Ai_singleton i
      exact le_antisymm this (Nat.succ_le_of_lt (_root_.pos_iff_ne_zero.mpr pos))
    have :
      ∀ r : ∀ i, α i, r ∈ piFinset A → (f fun i => g i (r i)) = f fun i => ∑ j ∈ A i, g i j := by
      intro r hr
      congr with i
      have : ∀ j ∈ A i, g i j = g i (r i) := by
        intro j hj
        congr
        apply Finset.card_le_one_iff.1 (Ai_singleton i) hj
        exact mem_piFinset.mp hr i
      simp only [Finset.sum_congr rfl this, Finset.mem_univ, Finset.sum_const, Ai_card i, one_nsmul]
    simp only [Finset.sum_congr rfl this, Ai_card, card_piFinset, prod_const_one, one_nsmul,
      Finset.sum_const]
  -- Remains the interesting case where one of the `A i`, say `A i₀`, has cardinality at least 2.
  -- We will split into two parts `B i₀` and `C i₀` of smaller cardinality, let `B i = C i = A i`
  -- for `i ≠ i₀`, apply the inductive assumption to `B` and `C`, and add up the corresponding
  -- parts to get the sum for `A`.
  /-
    case neg
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    Ai_singleton : Not (∀ (i : ι), LE.le (A i).card 1)
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  push_neg at Ai_singleton
  /-
    case neg
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    Ai_singleton : Exists fun i => LT.lt 1 (A i).card
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  obtain ⟨i₀, hi₀⟩ : ∃ i, 1 < #(A i) := Ai_singleton
  obtain ⟨j₁, j₂, _, hj₂, _⟩ : ∃ j₁ j₂, j₁ ∈ A i₀ ∧ j₂ ∈ A i₀ ∧ j₁ ≠ j₂ :=
    Finset.one_lt_card_iff.1 hi₀
  /-
    case neg.intro.intro.intro.intro.intro
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    i₀ : ι
    hi₀ : LT.lt 1 (A i₀).card
    j₁ j₂ : α i₀
    left✝ : Membership.mem (A i₀) j₁
    hj₂ : Membership.mem (A i₀) j₂
    right✝ : Ne j₁ j₂
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  let B := Function.update A i₀ (A i₀ \ {j₂})
  /-
    case neg.intro.intro.intro.intro.intro
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    i₀ : ι
    hi₀ : LT.lt 1 (A i₀).card
    j₁ j₂ : α i₀
    left✝ : Membership.mem (A i₀) j₁
    hj₂ : Membership.mem (A i₀) j₂
    right✝ : Ne j₁ j₂
    B : (a : ι) → Finset (α a) := Function.update A i₀ (SDiff.sdiff (A i₀) (Single …
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  let C := Function.update A i₀ {j₂}
  have B_subset_A : ∀ i, B i ⊆ A i := by
    intro i
    by_cases hi : i = i₀
    · rw [hi]
      simp only [B, sdiff_subset, update_self]
    · simp only [B, hi, update_of_ne, Ne, not_false_iff, Finset.Subset.refl]
  have C_subset_A : ∀ i, C i ⊆ A i := by
    intro i
    by_cases hi : i = i₀
    · rw [hi]
      simp only [C, hj₂, Finset.singleton_subset_iff, update_self]
    · simp only [C, hi, update_of_ne, Ne, not_false_iff, Finset.Subset.refl]
  -- split the sum at `i₀` as the sum over `B i₀` plus the sum over `C i₀`, to use additivity.
  have A_eq_BC :
    (fun i => ∑ j ∈ A i, g i j) =
      Function.update (fun i => ∑ j ∈ A i, g i j) i₀
        ((∑ j ∈ B i₀, g i₀ j) + ∑ j ∈ C i₀, g i₀ j) := by
    ext i
    by_cases hi : i = i₀
    · rw [hi, update_self]
      have : A i₀ = B i₀ ∪ C i₀ := by
        simp only [B, C, Function.update_self, Finset.sdiff_union_self_eq_union]
        symm
        simp only [hj₂, Finset.singleton_subset_iff, Finset.union_eq_left]
      rw [this]
      refine Finset.sum_union <| Finset.disjoint_right.2 fun j hj => ?_
      have : j = j₂ := by
        simpa [C] using hj
      rw [this]
      simp only [B, mem_sdiff, eq_self_iff_true, not_true, not_false_iff, Finset.mem_singleton,
        update_self, and_false]
    · simp [hi]
  have Beq :
    Function.update (fun i => ∑ j ∈ A i, g i j) i₀ (∑ j ∈ B i₀, g i₀ j) = fun i =>
      ∑ j ∈ B i, g i j := by
    ext i
    by_cases hi : i = i₀
    · rw [hi]
      simp only [update_self]
    · simp only [B, hi, update_of_ne, Ne, not_false_iff]
  have Ceq :
    Function.update (fun i => ∑ j ∈ A i, g i j) i₀ (∑ j ∈ C i₀, g i₀ j) = fun i =>
      ∑ j ∈ C i, g i j := by
    ext i
    by_cases hi : i = i₀
    · rw [hi]
      simp only [update_self]
    · simp only [C, hi, update_of_ne, Ne, not_false_iff]
  -- Express the inductive assumption for `B`
  have Brec : (f fun i => ∑ j ∈ B i, g i j) = ∑ r ∈ piFinset B, f fun i => g i (r i) := by
    have : ∑ i, #(B i) < ∑ i, #(A i) := by
      refine sum_lt_sum (fun i _ => card_le_card (B_subset_A i)) ⟨i₀, mem_univ _, ?_⟩
      have : {j₂} ⊆ A i₀ := by simp [hj₂]
      simp only [B, Finset.card_sdiff this, Function.update_self, Finset.card_singleton]
      exact Nat.pred_lt (ne_of_gt (lt_trans Nat.zero_lt_one hi₀))
    rw [h] at this
    exact IH _ this B rfl
  -- Express the inductive assumption for `C`
  have Crec : (f fun i => ∑ j ∈ C i, g i j) = ∑ r ∈ piFinset C, f fun i => g i (r i) := by
    have : (∑ i, #(C i)) < ∑ i, #(A i) :=
      Finset.sum_lt_sum (fun i _ => Finset.card_le_card (C_subset_A i))
        ⟨i₀, Finset.mem_univ _, by simp [C, hi₀]⟩
    rw [h] at this
    exact IH _ this C rfl
  have D : Disjoint (piFinset B) (piFinset C) :=
    haveI : Disjoint (B i₀) (C i₀) := by simp [B, C]
    piFinset_disjoint_of_disjoint B C this
  have pi_BC : piFinset A = piFinset B ∪ piFinset C := by
    apply Finset.Subset.antisymm
    · intro r hr
      by_cases hri₀ : r i₀ = j₂
      · apply Finset.mem_union_right
        refine mem_piFinset.2 fun i => ?_
        by_cases hi : i = i₀
        · have : r i₀ ∈ C i₀ := by simp [C, hri₀]
          rwa [hi]
        · simp [C, hi, mem_piFinset.1 hr i]
      · apply Finset.mem_union_left
        refine mem_piFinset.2 fun i => ?_
        by_cases hi : i = i₀
        · have : r i₀ ∈ B i₀ := by simp [B, hri₀, mem_piFinset.1 hr i₀]
          rwa [hi]
        · simp [B, hi, mem_piFinset.1 hr i]
    · exact
        Finset.union_subset (piFinset_subset _ _ fun i => B_subset_A i)
          (piFinset_subset _ _ fun i => C_subset_A i)
  /-
    case neg.intro.intro.intro.intro.intro
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    i₀ : ι
    hi₀ : LT.lt 1 (A i₀).card
    j₁ j₂ : α i₀
    left✝ : Membership.mem (A i₀) j₁
    hj₂ : Membership.mem (A i₀) j₂
    right✝ : Ne j₁ j₂
    B : (a : ι) → Finset (α a) := Function.update A i₀ (SDiff.sdiff (A i₀) (Single …
    C : (a : ι) → Finset (α a) := Function.update A i₀ (Singleton.singleton j₂)
    B_subset_A : ∀ (i : ι), HasSubset.Subset (B i) (A i)
    C_subset_A : ∀ (i : ι), HasSubset.Subset (C i) (A i)
    A_eq_BC : Eq (fun i => (A i).sum fun j => g i j) (Function.update (fun i => (A …
    Beq : Eq (Function.update (fun i => (A i).sum fun j => g i j) i₀ ((B i₀).sum f …
    Ceq : Eq (Function.update (fun i => (A i).sum fun j => g i j) i₀ ((C i₀).sum f …
    Brec : Eq (f fun i => (B i).sum fun j => g i j) ((Fintype.piFinset B).sum fun  …
    Crec : Eq (f fun i => (C i).sum fun j => g i j) ((Fintype.piFinset C).sum fun  …
    D : Disjoint (Fintype.piFinset B) (Fintype.piFinset C)
    pi_BC : Eq (Fintype.piFinset A) (Union.union (Fintype.piFinset B) (Fintype.piF …
    ⊢ Eq (f fun i => (A i).sum fun j => g i j) ((Fintype.piFinset A).sum fun r =>  …
  -/
  rw [A_eq_BC]
  /-
    case neg.intro.intro.intro.intro.intro
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    i₀ : ι
    hi₀ : LT.lt 1 (A i₀).card
    j₁ j₂ : α i₀
    left✝ : Membership.mem (A i₀) j₁
    hj₂ : Membership.mem (A i₀) j₂
    right✝ : Ne j₁ j₂
    B : (a : ι) → Finset (α a) := Function.update A i₀ (SDiff.sdiff (A i₀) (Single …
    C : (a : ι) → Finset (α a) := Function.update A i₀ (Singleton.singleton j₂)
    B_subset_A : ∀ (i : ι), HasSubset.Subset (B i) (A i)
    C_subset_A : ∀ (i : ι), HasSubset.Subset (C i) (A i)
    A_eq_BC : Eq (fun i => (A i).sum fun j => g i j) (Function.update (fun i => (A …
    Beq : Eq (Function.update (fun i => (A i).sum fun j => g i j) i₀ ((B i₀).sum f …
    Ceq : Eq (Function.update (fun i => (A i).sum fun j => g i j) i₀ ((C i₀).sum f …
    Brec : Eq (f fun i => (B i).sum fun j => g i j) ((Fintype.piFinset B).sum fun  …
    Crec : Eq (f fun i => (C i).sum fun j => g i j) ((Fintype.piFinset C).sum fun  …
    D : Disjoint (Fintype.piFinset B) (Fintype.piFinset C)
    pi_BC : Eq (Fintype.piFinset A) (Union.union (Fintype.piFinset B) (Fintype.piF …
    ⊢ Eq (f (Function.update (fun i => (A i).sum fun j => g i j) i₀ (HAdd.hAdd ((B …
  -/
  simp only [MultilinearMap.map_update_add, Beq, Ceq, Brec, Crec, pi_BC]
  /-
    case neg.intro.intro.intro.intro.intro
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    α : ι → Type u_1
    g : (i : ι) → α i → M₁ i
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    this : (i : ι) → DecidableEq (α i) := fun i => Classical.decEq (α i)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (A : (i : ι) → Finset (α i)), Eq (Finset.univ. …
    A : (i : ι) → Finset (α i)
    h : Eq (Finset.univ.sum fun i => (A i).card) n
    Ai_empty : ∀ (i : ι), Ne (A i) EmptyCollection.emptyCollection
    i₀ : ι
    hi₀ : LT.lt 1 (A i₀).card
    j₁ j₂ : α i₀
    left✝ : Membership.mem (A i₀) j₁
    hj₂ : Membership.mem (A i₀) j₂
    right✝ : Ne j₁ j₂
    B : (a : ι) → Finset (α a) := Function.update A i₀ (SDiff.sdiff (A i₀) (Single …
    C : (a : ι) → Finset (α a) := Function.update A i₀ (Singleton.singleton j₂)
    B_subset_A : ∀ (i : ι), HasSubset.Subset (B i) (A i)
    C_subset_A : ∀ (i : ι), HasSubset.Subset (C i) (A i)
    A_eq_BC : Eq (fun i => (A i).sum fun j => g i j) (Function.update (fun i => (A …
    Beq : Eq (Function.update (fun i => (A i).sum fun j => g i j) i₀ ((B i₀).sum f …
    Ceq : Eq (Function.update (fun i => (A i).sum fun j => g i j) i₀ ((C i₀).sum f …
    Brec : Eq (f fun i => (B i).sum fun j => g i j) ((Fintype.piFinset B).sum fun  …
    Crec : Eq (f fun i => (C i).sum fun j => g i j) ((Fintype.piFinset C).sum fun  …
    D : Disjoint (Fintype.piFinset B) (Fintype.piFinset C)
    pi_BC : Eq (Fintype.piFinset A) (Union.union (Fintype.piFinset B) (Fintype.piF …
    ⊢ Eq (HAdd.hAdd ((Fintype.piFinset B).sum fun r => f fun i => g i (r i)) ((Fin …
  -/
  rw [← Finset.sum_union D]
  /-
    🎉 no goals
  -/


/-- If `f` is multilinear, then `f (Σ_{j₁ ∈ A₁} g₁ j₁, ..., Σ_{jₙ ∈ Aₙ} gₙ jₙ)` is the sum of
`f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions with `r 1 ∈ A₁`, ...,
`r n ∈ Aₙ`. This follows from multilinearity by expanding successively with respect to each
coordinate. -/
theorem map_sum_finset [DecidableEq ι] [Fintype ι] :
    (f fun i => ∑ j ∈ A i, g i j) = ∑ r ∈ piFinset A, f fun i => g i (r i) :=
  f.map_sum_finset_aux _ _ rfl


/-- If `f` is multilinear, then `f (Σ_{j₁} g₁ j₁, ..., Σ_{jₙ} gₙ jₙ)` is the sum of
`f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions `r`. This follows from
multilinearity by expanding successively with respect to each coordinate. -/
theorem map_sum [DecidableEq ι] [Fintype ι] [∀ i, Fintype (α i)] :
    (f fun i => ∑ j, g i j) = ∑ r : ∀ i, α i, f fun i => g i (r i) :=
  f.map_sum_finset g fun _ => Finset.univ


theorem map_update_sum {α : Type*} [DecidableEq ι] (t : Finset α) (i : ι) (g : α → M₁ i)
    (m : ∀ i, M₁ i) : f (update m i (∑ a ∈ t, g a)) = ∑ a ∈ t, f (update m i (g a)) := by
  classical
    induction' t using Finset.induction with a t has ih h
    · simp
    · simp [Finset.sum_insert has, ih]


/-- Restrict the codomain of a multilinear map to a submodule.

This is the multilinear version of `LinearMap.codRestrict`. -/
@[simps]
def codRestrict (f : MultilinearMap R M₁ M₂) (p : Submodule R M₂) (h : ∀ v, f v ∈ p) :
    MultilinearMap R M₁ p where
  toFun v := ⟨f v, h v⟩
  map_update_add' _ _ _ _ := Subtype.ext <| MultilinearMap.map_update_add _ _ _ _ _
  map_update_smul' _ _ _ _ := Subtype.ext <| MultilinearMap.map_update_smul _ _ _ _ _


/-- Reinterpret an `A`-multilinear map as an `R`-multilinear map, if `A` is an algebra over `R`
and their actions on all involved modules agree with the action of `R` on `A`. -/
def restrictScalars (f : MultilinearMap A M₁ M₂) : MultilinearMap R M₁ M₂ where
  toFun := f
  map_update_add' := f.map_update_add
  map_update_smul' m i := (f.toLinearMap m i).map_smul_of_tower


@[simp]
theorem coe_restrictScalars (f : MultilinearMap A M₁ M₂) : ⇑(f.restrictScalars R) = f :=
  rfl


/-- Transfer the arguments to a map along an equivalence between argument indices.

The naming is derived from `Finsupp.domCongr`, noting that here the permutation applies to the
domain of the domain. -/
@[simps apply]
def domDomCongr (σ : ι₁ ≃ ι₂) (m : MultilinearMap R (fun _ : ι₁ => M₂) M₃) :
    MultilinearMap R (fun _ : ι₂ => M₂) M₃ where
  toFun v := m fun i => v (σ i)
  map_update_add' v i a b := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      inst✝ : DecidableEq ι₂
      v : ι₂ → M₂
      i : ι₂
      a b : M₂
      ⊢ Eq ((fun v => m fun i => v (σ i)) (Function.update v i (HAdd.hAdd a b))) (HA …
    -/
    letI := σ.injective.decidableEq
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      inst✝ : DecidableEq ι₂
      v : ι₂ → M₂
      i : ι₂
      a b : M₂
      this : DecidableEq ι₁ := ⋯.decidableEq
      ⊢ Eq ((fun v => m fun i => v (σ i)) (Function.update v i (HAdd.hAdd a b))) (HA …
    -/
    simp_rw [Function.update_apply_equiv_apply v]
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      inst✝ : DecidableEq ι₂
      v : ι₂ → M₂
      i : ι₂
      a b : M₂
      this : DecidableEq ι₁ := ⋯.decidableEq
      ⊢ Eq (m fun i_1 => Function.update (Function.comp v ⇑σ) (σ.symm i) (HAdd.hAdd  …
    -/
    rw [m.map_update_add]
    /-
      🎉 no goals
    -/
  map_update_smul' v i a b := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      inst✝ : DecidableEq ι₂
      v : ι₂ → M₂
      i : ι₂
      a : R
      b : M₂
      ⊢ Eq ((fun v => m fun i => v (σ i)) (Function.update v i (HSMul.hSMul a b))) ( …
    -/
    letI := σ.injective.decidableEq
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      inst✝ : DecidableEq ι₂
      v : ι₂ → M₂
      i : ι₂
      a : R
      b : M₂
      this : DecidableEq ι₁ := ⋯.decidableEq
      ⊢ Eq ((fun v => m fun i => v (σ i)) (Function.update v i (HSMul.hSMul a b))) ( …
    -/
    simp_rw [Function.update_apply_equiv_apply v]
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : AddCommMonoid M₃
      inst✝⁶ : AddCommMonoid M'
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      inst✝² : Module R M₃
      inst✝¹ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      inst✝ : DecidableEq ι₂
      v : ι₂ → M₂
      i : ι₂
      a : R
      b : M₂
      this : DecidableEq ι₁ := ⋯.decidableEq
      ⊢ Eq (m fun i_1 => Function.update (Function.comp v ⇑σ) (σ.symm i) (HSMul.hSMu …
    -/
    rw [m.map_update_smul]
    /-
      🎉 no goals
    -/


theorem domDomCongr_trans (σ₁ : ι₁ ≃ ι₂) (σ₂ : ι₂ ≃ ι₃)
    (m : MultilinearMap R (fun _ : ι₁ => M₂) M₃) :
    m.domDomCongr (σ₁.trans σ₂) = (m.domDomCongr σ₁).domDomCongr σ₂ :=
  rfl


theorem domDomCongr_mul (σ₁ : Equiv.Perm ι₁) (σ₂ : Equiv.Perm ι₁)
    (m : MultilinearMap R (fun _ : ι₁ => M₂) M₃) :
    m.domDomCongr (σ₂ * σ₁) = (m.domDomCongr σ₁).domDomCongr σ₂ :=
  rfl


/-- `MultilinearMap.domDomCongr` as an equivalence.

This is declared separately because it does not work with dot notation. -/
@[simps apply symm_apply]
def domDomCongrEquiv (σ : ι₁ ≃ ι₂) :
    MultilinearMap R (fun _ : ι₁ => M₂) M₃ ≃+ MultilinearMap R (fun _ : ι₂ => M₂) M₃ where
  toFun := domDomCongr σ
  invFun := domDomCongr σ.symm
  left_inv m := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁰ : Semiring R
      inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      ⊢ Eq (MultilinearMap.domDomCongr σ.symm (MultilinearMap.domDomCongr σ m)) m
    -/
    ext
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁰ : Semiring R
      inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      x✝ : ι₁ → M₂
      ⊢ Eq ((MultilinearMap.domDomCongr σ.symm (MultilinearMap.domDomCongr σ m)) x✝) …
    -/
    simp [domDomCongr]
    /-
      🎉 no goals
    -/
  right_inv m := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁰ : Semiring R
      inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      ⊢ Eq (MultilinearMap.domDomCongr σ (MultilinearMap.domDomCongr σ.symm m)) m
    -/
    ext
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁰ : Semiring R
      inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      m : MultilinearMap R (fun x => M₂) M₃
      x✝ : ι₂ → M₂
      ⊢ Eq ((MultilinearMap.domDomCongr σ (MultilinearMap.domDomCongr σ.symm m)) x✝) …
    -/
    simp [domDomCongr]
    /-
      🎉 no goals
    -/
  map_add' a b := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁰ : Semiring R
      inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      a b : MultilinearMap R (fun x => M₂) M₃
      ⊢ Eq ({ toFun := MultilinearMap.domDomCongr σ, invFun := MultilinearMap.domDom …
    -/
    ext
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹⁰ : Semiring R
      inst✝⁹ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : AddCommMonoid M₃
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      inst✝¹ : Module R M₃
      inst✝ : Module R M'
      f f' : MultilinearMap R M₁ M₂
      ι₁ : Type u_1
      ι₂ : Type u_2
      ι₃ : Type u_3
      σ : Equiv ι₁ ι₂
      a b : MultilinearMap R (fun x => M₂) M₃
      x✝ : ι₂ → M₂
      ⊢ Eq (({ toFun := MultilinearMap.domDomCongr σ, invFun := MultilinearMap.domDo …
    -/
    simp [domDomCongr]
    /-
      🎉 no goals
    -/


/-- The results of applying `domDomCongr` to two maps are equal if
and only if those maps are. -/
@[simp]
theorem domDomCongr_eq_iff (σ : ι₁ ≃ ι₂) (f g : MultilinearMap R (fun _ : ι₁ => M₂) M₃) :
    f.domDomCongr σ = g.domDomCongr σ ↔ f = g :=
  (domDomCongrEquiv σ : _ ≃+ MultilinearMap R (fun _ => M₂) M₃).apply_eq_iff_eq


lemma domDomRestrict_aux {ι} [DecidableEq ι] (P : ι → Prop) [DecidablePred P] {M₁ : ι → Type*}
    [DecidableEq {a // P a}]
    (x : (i : {a // P a}) → M₁ i) (z : (i : {a // ¬ P a}) → M₁ i) (i : {a : ι // P a})
    (c : M₁ i) : (fun j ↦ if h : P j then Function.update x i c ⟨j, h⟩ else z ⟨j, h⟩) =
    Function.update (fun j => if h : P j then x ⟨j, h⟩ else z ⟨j, h⟩) i c := by
  /-
    ι : Sort u_2
    inst✝² : DecidableEq ι
    P : ι → Prop
    inst✝¹ : DecidablePred P
    M₁ : ι → Type u_1
    inst✝ : DecidableEq (Subtype fun a => P a)
    x : (i : Subtype fun a => P a) → M₁ ↑i
    z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
    i : Subtype fun a => P a
    c : M₁ ↑i
    ⊢ Eq (fun j => dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨ …
  -/
  ext j
  /-
    case h
    ι : Sort u_2
    inst✝² : DecidableEq ι
    P : ι → Prop
    inst✝¹ : DecidablePred P
    M₁ : ι → Type u_1
    inst✝ : DecidableEq (Subtype fun a => P a)
    x : (i : Subtype fun a => P a) → M₁ ↑i
    z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
    i : Subtype fun a => P a
    c : M₁ ↑i
    j : ι
    ⊢ Eq (dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨j, h⟩) (F …
  -/
  by_cases h : j = i
    /-
      case pos
      ι : Sort u_2
      inst✝² : DecidableEq ι
      P : ι → Prop
      inst✝¹ : DecidablePred P
      M₁ : ι → Type u_1
      inst✝ : DecidableEq (Subtype fun a => P a)
      x : (i : Subtype fun a => P a) → M₁ ↑i
      z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
      i : Subtype fun a => P a
      c : M₁ ↑i
      j : ι
      h : Eq j ↑i
      ⊢ Eq (dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨j, h⟩) (F …
    -/
  · rw [h, Function.update_self]
    /-
      case pos
      ι : Sort u_2
      inst✝² : DecidableEq ι
      P : ι → Prop
      inst✝¹ : DecidablePred P
      M₁ : ι → Type u_1
      inst✝ : DecidableEq (Subtype fun a => P a)
      x : (i : Subtype fun a => P a) → M₁ ↑i
      z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
      i : Subtype fun a => P a
      c : M₁ ↑i
      j : ι
      h : Eq j ↑i
      ⊢ Eq (dite (P ↑i) (fun h => Function.update x i c ⟨↑i, h⟩) fun h => z ⟨↑i, h⟩) c
    -/
    simp only [i.2, update_self, dite_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Sort u_2
      inst✝² : DecidableEq ι
      P : ι → Prop
      inst✝¹ : DecidablePred P
      M₁ : ι → Type u_1
      inst✝ : DecidableEq (Subtype fun a => P a)
      x : (i : Subtype fun a => P a) → M₁ ↑i
      z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
      i : Subtype fun a => P a
      c : M₁ ↑i
      j : ι
      h : Not (Eq j ↑i)
      ⊢ Eq (dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨j, h⟩) (F …
    -/
  · rw [Function.update_of_ne h]
    /-
      case neg
      ι : Sort u_2
      inst✝² : DecidableEq ι
      P : ι → Prop
      inst✝¹ : DecidablePred P
      M₁ : ι → Type u_1
      inst✝ : DecidableEq (Subtype fun a => P a)
      x : (i : Subtype fun a => P a) → M₁ ↑i
      z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
      i : Subtype fun a => P a
      c : M₁ ↑i
      j : ι
      h : Not (Eq j ↑i)
      ⊢ Eq (dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨j, h⟩) (d …
    -/
    by_cases h' : P j
      /-
        case pos
        ι : Sort u_2
        inst✝² : DecidableEq ι
        P : ι → Prop
        inst✝¹ : DecidablePred P
        M₁ : ι → Type u_1
        inst✝ : DecidableEq (Subtype fun a => P a)
        x : (i : Subtype fun a => P a) → M₁ ↑i
        z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
        i : Subtype fun a => P a
        c : M₁ ↑i
        j : ι
        h : Not (Eq j ↑i)
        h' : P j
        ⊢ Eq (dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨j, h⟩) (d …
      -/
    · simp only [h', ne_eq, Subtype.mk.injEq, dite_true]
      have h'' : ¬ ⟨j, h'⟩ = i :=
        fun he => by apply_fun (fun x => x.1) at he; exact h he
      /-
        case pos
        ι : Sort u_2
        inst✝² : DecidableEq ι
        P : ι → Prop
        inst✝¹ : DecidablePred P
        M₁ : ι → Type u_1
        inst✝ : DecidableEq (Subtype fun a => P a)
        x : (i : Subtype fun a => P a) → M₁ ↑i
        z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
        i : Subtype fun a => P a
        c : M₁ ↑i
        j : ι
        h : Not (Eq j ↑i)
        h' : P j
        h'' : Not (Eq ⟨j, h'⟩ i)
        ⊢ Eq (Function.update x i c ⟨j, ⋯⟩) (x ⟨j, ⋯⟩)
      -/
      rw [Function.update_of_ne h'']
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Sort u_2
        inst✝² : DecidableEq ι
        P : ι → Prop
        inst✝¹ : DecidablePred P
        M₁ : ι → Type u_1
        inst✝ : DecidableEq (Subtype fun a => P a)
        x : (i : Subtype fun a => P a) → M₁ ↑i
        z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
        i : Subtype fun a => P a
        c : M₁ ↑i
        j : ι
        h : Not (Eq j ↑i)
        h' : Not (P j)
        ⊢ Eq (dite (P j) (fun h => Function.update x i c ⟨j, h⟩) fun h => z ⟨j, h⟩) (d …
      -/
    · simp only [h', ne_eq, Subtype.mk.injEq, dite_false]
      /-
        🎉 no goals
      -/


lemma domDomRestrict_aux_right {ι} [DecidableEq ι] (P : ι → Prop) [DecidablePred P] {M₁ : ι → Type*}
    [DecidableEq {a // ¬ P a}]
    (x : (i : {a // P a}) → M₁ i) (z : (i : {a // ¬ P a}) → M₁ i) (i : {a : ι // ¬ P a})
    (c : M₁ i) : (fun j ↦ if h : P j then x ⟨j, h⟩ else Function.update z i c ⟨j, h⟩) =
    Function.update (fun j => if h : P j then x ⟨j, h⟩ else z ⟨j, h⟩) i c := by
  /-
    ι : Sort u_2
    inst✝² : DecidableEq ι
    P : ι → Prop
    inst✝¹ : DecidablePred P
    M₁ : ι → Type u_1
    inst✝ : DecidableEq (Subtype fun a => Not (P a))
    x : (i : Subtype fun a => P a) → M₁ ↑i
    z : (i : Subtype fun a => Not (P a)) → M₁ ↑i
    i : Subtype fun a => Not (P a)
    c : M₁ ↑i
    ⊢ Eq (fun j => dite (P j) (fun h => x ⟨j, h⟩) fun h => Function.update z i c ⟨ …
  -/
  simpa only [dite_not] using domDomRestrict_aux _ z (fun j ↦ x ⟨j.1, not_not.mp j.2⟩) i c
  /-
    🎉 no goals
  -/


/-- Given a multilinear map `f` on `(i : ι) → M i`, a (decidable) predicate `P` on `ι` and
an element `z` of `(i : {a // ¬ P a}) → M₁ i`, construct a multilinear map on
`(i : {a // P a}) → M₁ i)` whose value at `x` is `f` evaluated at the vector with `i`th coordinate
`x i` if `P i` and `z i` otherwise.

The naming is similar to `MultilinearMap.domDomCongr`: here we are applying the restriction to the
domain of the domain.

For a linear map version, see `MultilinearMap.domDomRestrictₗ`.
-/
def domDomRestrict (f : MultilinearMap R M₁ M₂) (P : ι → Prop) [DecidablePred P]
    (z : (i : {a : ι // ¬ P a}) → M₁ i) :
    MultilinearMap R (fun (i : {a : ι // P a}) => M₁ i) M₂ where
  toFun x := f (fun j ↦ if h : P j then x ⟨j, h⟩ else z ⟨j, h⟩)
  map_update_add' x i a b := by
    classical
    simp only
    repeat (rw [domDomRestrict_aux])
    simp only [MultilinearMap.map_update_add]
  map_update_smul' z i c a := by
    classical
    simp only
    repeat (rw [domDomRestrict_aux])
    simp only [MultilinearMap.map_update_smul]


@[simp]
lemma domDomRestrict_apply (f : MultilinearMap R M₁ M₂) (P : ι → Prop)
    [DecidablePred P] (x : (i : {a // P a}) → M₁ i) (z : (i : {a // ¬ P a}) → M₁ i) :
    f.domDomRestrict P z x = f (fun j => if h : P j then x ⟨j, h⟩ else z ⟨j, h⟩) := rfl

-- TODO: Should add a ref here when available.

/-- The "derivative" of a multilinear map, as a linear map from `(i : ι) → M₁ i` to `M₂`.
For continuous multilinear maps, this will indeed be the derivative. -/
def linearDeriv [DecidableEq ι] [Fintype ι] (f : MultilinearMap R M₁ M₂)
    (x : (i : ι) → M₁ i) : ((i : ι) → M₁ i) →ₗ[R] M₂ :=
  ∑ i : ι, (f.toLinearMap x i).comp (LinearMap.proj i)


@[simp]
lemma linearDeriv_apply [DecidableEq ι] [Fintype ι] (f : MultilinearMap R M₁ M₂)
    (x y : (i : ι) → M₁ i) :
    f.linearDeriv x y = ∑ i, f (update x i (y i)) := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : Semiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : MultilinearMap R M₁ M₂
    x y : (i : ι) → M₁ i
    ⊢ Eq ((f.linearDeriv x) y) (Finset.univ.sum fun i => f (Function.update x i (y …
  -/
  unfold linearDeriv
  simp only [LinearMap.coeFn_sum, LinearMap.coe_comp, LinearMap.coe_proj, Finset.sum_apply,
    Function.comp_apply, Function.eval, toLinearMap_apply]


/-- Composing a multilinear map with a linear map gives again a multilinear map. -/
def compMultilinearMap (g : M₂ →ₗ[R] M₃) (f : MultilinearMap R M₁ M₂) : MultilinearMap R M₁ M₃ where
  toFun := g ∘ f
                                /-
                                  R : Type uR
                                  S : Type uS
                                  ι : Type uι
                                  n : Nat
                                  M : Fin n.succ → Type v
                                  M₁ : ι → Type v₁
                                  M₂ : Type v₂
                                  M₃ : Type v₃
                                  M' : Type v'
                                  inst✝⁹ : Semiring R
                                  inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                                  inst✝⁷ : AddCommMonoid M₂
                                  inst✝⁶ : AddCommMonoid M₃
                                  inst✝⁵ : AddCommMonoid M'
                                  inst✝⁴ : (i : ι) → Module R (M₁ i)
                                  inst✝³ : Module R M₂
                                  inst✝² : Module R M₃
                                  inst✝¹ : Module R M'
                                  g : LinearMap (RingHom.id R) M₂ M₃
                                  f : MultilinearMap R M₁ M₂
                                  inst✝ : DecidableEq ι
                                  m : (i : ι) → M₁ i
                                  i : ι
                                  x y : M₁ i
                                  ⊢ Eq (Function.comp (⇑g) (⇑f) (Function.update m i (HAdd.hAdd x y))) (HAdd.hAd …
                                -/
  map_update_add' m i x y := by simp
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   R : Type uR
                                   S : Type uS
                                   ι : Type uι
                                   n : Nat
                                   M : Fin n.succ → Type v
                                   M₁ : ι → Type v₁
                                   M₂ : Type v₂
                                   M₃ : Type v₃
                                   M' : Type v'
                                   inst✝⁹ : Semiring R
                                   inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                                   inst✝⁷ : AddCommMonoid M₂
                                   inst✝⁶ : AddCommMonoid M₃
                                   inst✝⁵ : AddCommMonoid M'
                                   inst✝⁴ : (i : ι) → Module R (M₁ i)
                                   inst✝³ : Module R M₂
                                   inst✝² : Module R M₃
                                   inst✝¹ : Module R M'
                                   g : LinearMap (RingHom.id R) M₂ M₃
                                   f : MultilinearMap R M₁ M₂
                                   inst✝ : DecidableEq ι
                                   m : (i : ι) → M₁ i
                                   i : ι
                                   c : R
                                   x : M₁ i
                                   ⊢ Eq (Function.comp (⇑g) (⇑f) (Function.update m i (HSMul.hSMul c x))) (HSMul. …
                                 -/
  map_update_smul' m i c x := by simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem coe_compMultilinearMap (g : M₂ →ₗ[R] M₃) (f : MultilinearMap R M₁ M₂) :
    ⇑(g.compMultilinearMap f) = g ∘ f :=
  rfl


@[simp]
theorem compMultilinearMap_apply (g : M₂ →ₗ[R] M₃) (f : MultilinearMap R M₁ M₂) (m : ∀ i, M₁ i) :
    g.compMultilinearMap f m = g (f m) :=
  rfl


@[simp]
theorem compMultilinearMap_zero (g : M₂ →ₗ[R] M₃) :
    g.compMultilinearMap (0 : MultilinearMap R M₁ M₂) = 0 :=
  MultilinearMap.ext fun _ => map_zero g


@[simp]
theorem zero_compMultilinearMap (f: MultilinearMap R M₁ M₂) :
    (0 : M₂ →ₗ[R] M₃).compMultilinearMap f = 0 := rfl


@[simp]
theorem compMultilinearMap_add (g : M₂ →ₗ[R] M₃) (f₁ f₂ : MultilinearMap R M₁ M₂) :
    g.compMultilinearMap (f₁ + f₂) = g.compMultilinearMap f₁ + g.compMultilinearMap f₂ :=
  MultilinearMap.ext fun _ => map_add g _ _


@[simp]
theorem add_compMultilinearMap (g₁ g₂ : M₂ →ₗ[R] M₃) (f: MultilinearMap R M₁ M₂) :
    (g₁ + g₂).compMultilinearMap f = g₁.compMultilinearMap f + g₂.compMultilinearMap f := rfl


@[simp]
theorem compMultilinearMap_smul [Monoid S] [DistribMulAction S M₂] [DistribMulAction S M₃]
    [SMulCommClass R S M₂] [SMulCommClass R S M₃] [CompatibleSMul M₂ M₃ S R]
    (g : M₂ →ₗ[R] M₃) (s : S) (f : MultilinearMap R M₁ M₂) :
    g.compMultilinearMap (s • f) = s • g.compMultilinearMap f :=
  MultilinearMap.ext fun _ => g.map_smul_of_tower _ _


@[simp]
theorem smul_compMultilinearMap [Monoid S] [DistribMulAction S M₃] [SMulCommClass R S M₃]
    (g : M₂ →ₗ[R] M₃) (s : S) (f : MultilinearMap R M₁ M₂) :
    (s • g).compMultilinearMap f = s • g.compMultilinearMap f := rfl


/-- The multilinear version of `LinearMap.subtype_comp_codRestrict` -/
@[simp]
theorem subtype_compMultilinearMap_codRestrict (f : MultilinearMap R M₁ M₂) (p : Submodule R M₂)
    (h) : p.subtype.compMultilinearMap (f.codRestrict p h) = f :=
  rfl


/-- The multilinear version of `LinearMap.comp_codRestrict` -/
@[simp]
theorem compMultilinearMap_codRestrict (g : M₂ →ₗ[R] M₃) (f : MultilinearMap R M₁ M₂)
    (p : Submodule R M₃) (h) :
    (g.codRestrict p h).compMultilinearMap f =
      (g.compMultilinearMap f).codRestrict p fun v => h (f v) :=
  rfl


@[simp]
theorem compMultilinearMap_domDomCongr (σ : ι₁ ≃ ι₂) (g : M₂ →ₗ[R] M₃)
    (f : MultilinearMap R (fun _ : ι₁ => M') M₂) :
    (g.compMultilinearMap f).domDomCongr σ = g.compMultilinearMap (f.domDomCongr σ) := by
  /-
    R : Type uR
    M₂ : Type v₂
    M₃ : Type v₃
    M' : Type v'
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Module R M'
    ι₁ : Type u_1
    ι₂ : Type u_2
    σ : Equiv ι₁ ι₂
    g : LinearMap (RingHom.id R) M₂ M₃
    f : MultilinearMap R (fun x => M') M₂
    ⊢ Eq (MultilinearMap.domDomCongr σ (g.compMultilinearMap f)) (g.compMultilinea …
  -/
  ext
  /-
    case H
    R : Type uR
    M₂ : Type v₂
    M₃ : Type v₃
    M' : Type v'
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Module R M'
    ι₁ : Type u_1
    ι₂ : Type u_2
    σ : Equiv ι₁ ι₂
    g : LinearMap (RingHom.id R) M₂ M₃
    f : MultilinearMap R (fun x => M') M₂
    x✝ : ι₂ → M'
    ⊢ Eq ((MultilinearMap.domDomCongr σ (g.compMultilinearMap f)) x✝) ((g.compMult …
  -/
  simp [MultilinearMap.domDomCongr]
  /-
    🎉 no goals
  -/


instance [Monoid S] [DistribMulAction S M₂] [Module R M₂] [SMulCommClass R S M₂] :
    DistribMulAction S (MultilinearMap R M₁ M₂) :=
  coe_injective.distribMulAction coeAddMonoidHom fun _ _ ↦ rfl


/-- The space of multilinear maps over an algebra over `R` is a module over `R`, for the pointwise
addition and scalar multiplication. -/
instance : Module S (MultilinearMap R M₁ M₂) :=
  coe_injective.module _ coeAddMonoidHom fun _ _ ↦ rfl


instance [NoZeroSMulDivisors S M₂] : NoZeroSMulDivisors S (MultilinearMap R M₁ M₂) :=
  coe_injective.noZeroSMulDivisors _ rfl coe_smul


variable (S) in
/-- `LinearMap.compMultilinearMap` as an `S`-linear map. -/
@[simps]
def _root_.LinearMap.compMultilinearMapₗ [Semiring S] [Module S M₂] [Module S M₃]
    [SMulCommClass R S M₂] [SMulCommClass R S M₃] [LinearMap.CompatibleSMul M₂ M₃ S R]
    (g : M₂ →ₗ[R] M₃) :
    MultilinearMap R M₁ M₂ →ₗ[S] MultilinearMap R M₁ M₃ where
  toFun := g.compMultilinearMap
  map_add' := g.compMultilinearMap_add
  map_smul' := g.compMultilinearMap_smul


/-- Linear equivalence between linear maps `M₂ →ₗ[R] M₃`
and one-multilinear maps `MultilinearMap R (fun _ : ι ↦ M₂) M₃`. -/
@[simps (config := { simpRhs := true })]
def ofSubsingletonₗ [Subsingleton ι] (i : ι) :
    (M₂ →ₗ[R] M₃) ≃ₗ[S] MultilinearMap R (fun _ : ι ↦ M₂) M₃ :=
  { ofSubsingleton R M₂ M₃ i with
    map_add' := fun _ _ ↦ rfl
    map_smul' := fun _ _ ↦ rfl }


/-- The dependent version of `MultilinearMap.domDomCongrLinearEquiv`. -/
@[simps apply symm_apply]
def domDomCongrLinearEquiv' {ι' : Type*} (σ : ι ≃ ι') :
    MultilinearMap R M₁ M₂ ≃ₗ[S] MultilinearMap R (fun i => M₁ (σ.symm i)) M₂ where
  toFun f :=
    { toFun := f ∘ (σ.piCongrLeft' M₁).symm
      map_update_add' := fun m i => by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          ⊢ ∀ (x y : M₁ (σ.symm i)), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ). …
        -/
        letI := σ.decidableEq
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          this : DecidableEq ι := σ.decidableEq
          ⊢ ∀ (x y : M₁ (σ.symm i)), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ). …
        -/
        rw [← σ.apply_symm_apply i]
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          this : DecidableEq ι := σ.decidableEq
          ⊢ ∀ (x y : M₁ (σ.symm (σ (σ.symm i)))), Eq (Function.comp (⇑f) (⇑(Equiv.piCong …
        -/
        intro x y
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          this : DecidableEq ι := σ.decidableEq
          x y : M₁ (σ.symm (σ (σ.symm i)))
          ⊢ Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ).symm) (Function.update m  …
        -/
        simp only [comp_apply, piCongrLeft'_symm_update, f.map_update_add]
        /-
          🎉 no goals
        -/
      map_update_smul' := fun m i c => by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          c : R
          ⊢ ∀ (x : M₁ (σ.symm i)), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ).sy …
        -/
        letI := σ.decidableEq
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          c : R
          this : DecidableEq ι := σ.decidableEq
          ⊢ ∀ (x : M₁ (σ.symm i)), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ).sy …
        -/
        rw [← σ.apply_symm_apply i]
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          c : R
          this : DecidableEq ι := σ.decidableEq
          ⊢ ∀ (x : M₁ (σ.symm (σ (σ.symm i)))), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrL …
        -/
        intro x
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R M₁ M₂
          inst✝ : DecidableEq ι'
          m : (i : ι') → M₁ (σ.symm i)
          i : ι'
          c : R
          this : DecidableEq ι := σ.decidableEq
          x : M₁ (σ.symm (σ (σ.symm i)))
          ⊢ Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ).symm) (Function.update m  …
        -/
        simp only [Function.comp, piCongrLeft'_symm_update, f.map_update_smul] }
        /-
          🎉 no goals
        -/
  invFun f :=
    { toFun := f ∘ σ.piCongrLeft' M₁
      map_update_add' := fun m i => by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          ⊢ ∀ (x y : M₁ i), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ)) (Functio …
        -/
        letI := σ.symm.decidableEq
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          this : DecidableEq ι' := σ.symm.decidableEq
          ⊢ ∀ (x y : M₁ i), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ)) (Functio …
        -/
        rw [← σ.symm_apply_apply i]
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          this : DecidableEq ι' := σ.symm.decidableEq
          ⊢ ∀ (x y : M₁ (σ.symm (σ i))), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ …
        -/
        intro x y
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          this : DecidableEq ι' := σ.symm.decidableEq
          x y : M₁ (σ.symm (σ i))
          ⊢ Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ)) (Function.update m (σ.sy …
        -/
        simp only [comp_apply, piCongrLeft'_update, f.map_update_add]
        /-
          🎉 no goals
        -/
      map_update_smul' := fun m i c => by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      f₁ f₂ : MultilinearMap R M₁ M₂
      ⊢ Eq ((fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁ σ).symm, m …
    -/
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          c : R
          ⊢ ∀ (x : M₁ i), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ)) (Function. …
        -/
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      f₁ f₂ : MultilinearMap R M₁ M₂
      x✝ : (i : ι') → M₁ (σ.symm i)
      ⊢ Eq (((fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁ σ).symm,  …
    -/
        letI := σ.symm.decidableEq
    /-
      🎉 no goals
    -/
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          c : R
          this : DecidableEq ι' := σ.symm.decidableEq
          ⊢ ∀ (x : M₁ i), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ)) (Function. …
        -/
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      c : S
      f : MultilinearMap R M₁ M₂
      ⊢ Eq ({ toFun := fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁  …
    -/
        rw [← σ.symm_apply_apply i]
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      c : S
      f : MultilinearMap R M₁ M₂
      x✝ : (i : ι') → M₁ (σ.symm i)
      ⊢ Eq (({ toFun := fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁ …
    -/
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          c : R
          this : DecidableEq ι' := σ.symm.decidableEq
          ⊢ ∀ (x : M₁ (σ.symm (σ i))), Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ …
        -/
    /-
      🎉 no goals
    -/
        intro x
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝¹² : Semiring R
          inst✝¹¹ : (i : ι) → AddCommMonoid (M₁ i)
          inst✝¹⁰ : (i : ι) → Module R (M₁ i)
          inst✝⁹ : AddCommMonoid M₂
          inst✝⁸ : Module R M₂
          inst✝⁷ : Semiring S
          inst✝⁶ : Module S M₂
          inst✝⁵ : SMulCommClass R S M₂
          inst✝⁴ : AddCommMonoid M₃
          inst✝³ : Module S M₃
          inst✝² : Module R M₃
          inst✝¹ : SMulCommClass R S M₃
          ι' : Type u_1
          σ : Equiv ι ι'
          f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
          inst✝ : DecidableEq ι
          m : (i : ι) → M₁ i
          i : ι
          c : R
          this : DecidableEq ι' := σ.symm.decidableEq
          x : M₁ (σ.symm (σ i))
          ⊢ Eq (Function.comp (⇑f) (⇑(Equiv.piCongrLeft' M₁ σ)) (Function.update m (σ.sy …
        -/
        simp only [Function.comp, piCongrLeft'_update, f.map_update_smul] }
        /-
          🎉 no goals
        -/
  map_add' f₁ f₂ := by
    ext
    simp only [Function.comp, coe_mk, add_apply]
  map_smul' c f := by
    ext
    simp only [Function.comp, coe_mk, smul_apply, RingHom.id_apply]
  left_inv f := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      f : MultilinearMap R M₁ M₂
      ⊢ Eq ((fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁ σ), map_up …
    -/
    ext
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      f : MultilinearMap R M₁ M₂
      x✝ : (i : ι) → M₁ i
      ⊢ Eq (((fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁ σ), map_u …
    -/
    simp only [coe_mk, comp_apply, Equiv.symm_apply_apply]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
      ⊢ Eq ({ toFun := fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁  …
    -/
    ext
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝¹¹ : Semiring R
      inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁹ : (i : ι) → Module R (M₁ i)
      inst✝⁸ : AddCommMonoid M₂
      inst✝⁷ : Module R M₂
      inst✝⁶ : Semiring S
      inst✝⁵ : Module S M₂
      inst✝⁴ : SMulCommClass R S M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module S M₃
      inst✝¹ : Module R M₃
      inst✝ : SMulCommClass R S M₃
      ι' : Type u_1
      σ : Equiv ι ι'
      f : MultilinearMap R (fun i => M₁ (σ.symm i)) M₂
      x✝ : (i : ι') → M₁ (σ.symm i)
      ⊢ Eq (({ toFun := fun f => { toFun := Function.comp ⇑f ⇑(Equiv.piCongrLeft' M₁ …
    -/
    simp only [coe_mk, comp_apply, Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


/-- The space of constant maps is equivalent to the space of maps that are multilinear with respect
to an empty family. -/
@[simps]
def constLinearEquivOfIsEmpty [IsEmpty ι] : M₂ ≃ₗ[S] MultilinearMap R M₁ M₂ where
  toFun := MultilinearMap.constOfIsEmpty R _
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun f := f 0
  left_inv _ := rfl
  right_inv f := ext fun _ => MultilinearMap.congr_arg f <| Subsingleton.elim _ _


/-- `MultilinearMap.domDomCongr` as a `LinearEquiv`. -/
@[simps apply symm_apply]
def domDomCongrLinearEquiv {ι₁ ι₂} (σ : ι₁ ≃ ι₂) :
    MultilinearMap R (fun _ : ι₁ => M₂) M₃ ≃ₗ[S] MultilinearMap R (fun _ : ι₂ => M₂) M₃ :=
  { (domDomCongrEquiv σ :
      MultilinearMap R (fun _ : ι₁ => M₂) M₃ ≃+ MultilinearMap R (fun _ : ι₂ => M₂) M₃) with
    map_smul' := fun c f => by
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁹ : (i : ι) → Module R (M₁ i)
        inst✝⁸ : AddCommMonoid M₂
        inst✝⁷ : Module R M₂
        inst✝⁶ : Semiring S
        inst✝⁵ : Module S M₂
        inst✝⁴ : SMulCommClass R S M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : Module S M₃
        inst✝¹ : Module R M₃
        inst✝ : SMulCommClass R S M₃
        ι₁ : Type ?u.261772
        ι₂ : Type ?u.261891
        σ : Equiv ι₁ ι₂
        c : S
        f : MultilinearMap R (fun x => M₂) M₃
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) (HSMul …
      -/
      ext
      /-
        case H
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁹ : (i : ι) → Module R (M₁ i)
        inst✝⁸ : AddCommMonoid M₂
        inst✝⁷ : Module R M₂
        inst✝⁶ : Semiring S
        inst✝⁵ : Module S M₂
        inst✝⁴ : SMulCommClass R S M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : Module S M₃
        inst✝¹ : Module R M₃
        inst✝ : SMulCommClass R S M₃
        ι₁ : Type ?u.261772
        ι₂ : Type ?u.261891
        σ : Equiv ι₁ ι₂
        c : S
        f : MultilinearMap R (fun x => M₂) M₃
        x✝ : ι₂ → M₂
        ⊢ Eq (({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) x✝) ( …
      -/
      simp [MultilinearMap.domDomCongr] }
      /-
        🎉 no goals
      -/


/-- Given a predicate `P`, one may associate to a multilinear map `f` a multilinear map
from the elements satisfying `P` to the multilinear maps on elements not satisfying `P`.
In other words, splitting the variables into two subsets one gets a multilinear map into
multilinear maps.
This is a linear map version of the function `MultilinearMap.domDomRestrict`. -/
def domDomRestrictₗ (f : MultilinearMap R M₁ M₂) (P : ι → Prop) [DecidablePred P] :
    MultilinearMap R (fun (i : {a : ι // ¬ P a}) => M₁ i)
      (MultilinearMap R (fun (i : {a : ι // P a}) => M₁ i) M₂) where
  toFun := fun z ↦ domDomRestrict f P z
  map_update_add' := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      f : MultilinearMap R M₁ M₂
      P : ι → Prop
      inst✝ : DecidablePred P
      ⊢ ∀ [inst : DecidableEq (Subtype fun a => Not (P a))] (m : (i : Subtype fun a  …
    -/
    intro h m i x y
    classical
    ext v
    simp [domDomRestrict_aux_right]
  map_update_smul' := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      f : MultilinearMap R M₁ M₂
      P : ι → Prop
      inst✝ : DecidablePred P
      ⊢ ∀ [inst : DecidableEq (Subtype fun a => Not (P a))] (m : (i : Subtype fun a  …
    -/
    intro h m i c x
    classical
    ext v
    simp [domDomRestrict_aux_right]


lemma iteratedFDeriv_aux {ι} {M₁ : ι → Type*} {α : Type*} [DecidableEq α]
    (s : Set ι) [DecidableEq { x // x ∈ s }] (e : α ≃ s)
    (m : α → ((i : ι) → M₁ i)) (a : α) (z : (i : ι) → M₁ i) :
    (fun i ↦ update m a z (e.symm i) i) =
      (fun i ↦ update (fun j ↦ m (e.symm j) j) (e a) (z (e a)) i) := by
  /-
    ι : Type u_4
    M₁ : ι → Type u_2
    α : Type u_3
    inst✝¹ : DecidableEq α
    s : Set ι
    inst✝ : DecidableEq (Subtype fun x => Membership.mem s x)
    e : Equiv α ↑s
    m : α → (i : ι) → M₁ i
    a : α
    z : (i : ι) → M₁ i
    ⊢ Eq (fun i => Function.update m a z (e.symm i) ↑i) fun i => Function.update ( …
  -/
  ext i
  /-
    case h
    ι : Type u_4
    M₁ : ι → Type u_2
    α : Type u_3
    inst✝¹ : DecidableEq α
    s : Set ι
    inst✝ : DecidableEq (Subtype fun x => Membership.mem s x)
    e : Equiv α ↑s
    m : α → (i : ι) → M₁ i
    a : α
    z : (i : ι) → M₁ i
    i : ↑s
    ⊢ Eq (Function.update m a z (e.symm i) ↑i) (Function.update (fun j => m (e.sym …
  -/
  rcases eq_or_ne a (e.symm i) with rfl | hne
    /-
      case h.inl
      ι : Type u_4
      M₁ : ι → Type u_2
      α : Type u_3
      inst✝¹ : DecidableEq α
      s : Set ι
      inst✝ : DecidableEq (Subtype fun x => Membership.mem s x)
      e : Equiv α ↑s
      m : α → (i : ι) → M₁ i
      z : (i : ι) → M₁ i
      i : ↑s
      ⊢ Eq (Function.update m (e.symm i) z (e.symm i) ↑i) (Function.update (fun j => …
    -/
  · rw [Equiv.apply_symm_apply e i, update_self, update_self]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u_4
      M₁ : ι → Type u_2
      α : Type u_3
      inst✝¹ : DecidableEq α
      s : Set ι
      inst✝ : DecidableEq (Subtype fun x => Membership.mem s x)
      e : Equiv α ↑s
      m : α → (i : ι) → M₁ i
      a : α
      z : (i : ι) → M₁ i
      i : ↑s
      hne : Ne a (e.symm i)
      ⊢ Eq (Function.update m a z (e.symm i) ↑i) (Function.update (fun j => m (e.sym …
    -/
  · rw [update_of_ne hne.symm, update_of_ne fun h ↦ (Equiv.symm_apply_apply .. ▸ h ▸ hne) rfl]
    /-
      🎉 no goals
    -/


/-- One of the components of the iterated derivative of a multilinear map. Given a bijection `e`
between a type `α` (typically `Fin k`) and a subset `s` of `ι`, this component is a multilinear map
of `k` vectors `v₁, ..., vₖ`, mapping them to `f (x₁, (v_{e.symm 2})₂, x₃, ...)`, where at
indices `i` in `s` one uses the `i`-th coordinate of the vector `v_{e.symm i}` and otherwise one
uses the `i`-th coordinate of a reference vector `x`.
This is multilinear in the components of `x` outside of `s`, and in the `v_j`. -/
noncomputable def iteratedFDerivComponent {α : Type*}
    (f : MultilinearMap R M₁ M₂) {s : Set ι} (e : α ≃ s) [DecidablePred (· ∈ s)] :
    MultilinearMap R (fun (i : {a : ι // a ∉ s}) ↦ M₁ i)
      (MultilinearMap R (fun (_ : α) ↦ (∀ i, M₁ i)) M₂) where
  toFun := fun z ↦
    { toFun := fun v ↦ domDomRestrictₗ f (fun i ↦ i ∈ s) z (fun i ↦ v (e.symm i) i)
                            /-
                              R : Type uR
                              S : Type uS
                              ι : Type uι
                              n : Nat
                              M : Fin n.succ → Type v
                              M₁ : ι → Type v₁
                              M₂ : Type v₂
                              M₃ : Type v₃
                              M' : Type v'
                              inst✝⁹ : CommSemiring R
                              inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                              inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
                              inst✝⁶ : AddCommMonoid M₂
                              inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                              inst✝⁴ : (i : ι) → Module R (M₁ i)
                              inst✝³ : Module R M₂
                              f✝ f' : MultilinearMap R M₁ M₂
                              M₁' : ι → Type u_1
                              inst✝² : (i : ι) → AddCommMonoid (M₁' i)
                              inst✝¹ : (i : ι) → Module R (M₁' i)
                              α : Type u_2
                              f : MultilinearMap R M₁ M₂
                              s : Set ι
                              e : Equiv α ↑s
                              inst✝ : DecidablePred fun x => Membership.mem s x
                              z : (i : Subtype fun a => Not (Membership.mem s a)) → M₁ ↑i
                              ⊢ ∀ [inst : DecidableEq α] (m : α → (i : ι) → M₁ i) (i : α) (x y : (i : ι) → M …
                            -/
      map_update_add' := by classical simp [iteratedFDeriv_aux]
                            /-
                              🎉 no goals
                            -/
                             /-
                               R : Type uR
                               S : Type uS
                               ι : Type uι
                               n : Nat
                               M : Fin n.succ → Type v
                               M₁ : ι → Type v₁
                               M₂ : Type v₂
                               M₃ : Type v₃
                               M' : Type v'
                               inst✝⁹ : CommSemiring R
                               inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                               inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
                               inst✝⁶ : AddCommMonoid M₂
                               inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                               inst✝⁴ : (i : ι) → Module R (M₁ i)
                               inst✝³ : Module R M₂
                               f✝ f' : MultilinearMap R M₁ M₂
                               M₁' : ι → Type u_1
                               inst✝² : (i : ι) → AddCommMonoid (M₁' i)
                               inst✝¹ : (i : ι) → Module R (M₁' i)
                               α : Type u_2
                               f : MultilinearMap R M₁ M₂
                               s : Set ι
                               e : Equiv α ↑s
                               inst✝ : DecidablePred fun x => Membership.mem s x
                               z : (i : Subtype fun a => Not (Membership.mem s a)) → M₁ ↑i
                               ⊢ ∀ [inst : DecidableEq α] (m : α → (i : ι) → M₁ i) (i : α) (c : R) (x : (i :  …
                             -/
      map_update_smul' := by classical simp [iteratedFDeriv_aux] }
                             /-
                               🎉 no goals
                             -/
                        /-
                          R : Type uR
                          S : Type uS
                          ι : Type uι
                          n : Nat
                          M : Fin n.succ → Type v
                          M₁ : ι → Type v₁
                          M₂ : Type v₂
                          M₃ : Type v₃
                          M' : Type v'
                          inst✝⁹ : CommSemiring R
                          inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                          inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
                          inst✝⁶ : AddCommMonoid M₂
                          inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                          inst✝⁴ : (i : ι) → Module R (M₁ i)
                          inst✝³ : Module R M₂
                          f✝ f' : MultilinearMap R M₁ M₂
                          M₁' : ι → Type u_1
                          inst✝² : (i : ι) → AddCommMonoid (M₁' i)
                          inst✝¹ : (i : ι) → Module R (M₁' i)
                          α : Type u_2
                          f : MultilinearMap R M₁ M₂
                          s : Set ι
                          e : Equiv α ↑s
                          inst✝ : DecidablePred fun x => Membership.mem s x
                          ⊢ ∀ [inst : DecidableEq (Subtype fun a => Not (Membership.mem s a))] (m : (i : …
                        -/
  map_update_add' := by intros; ext; simp
                                     /-
                                       🎉 no goals
                                     -/
                         /-
                           R : Type uR
                           S : Type uS
                           ι : Type uι
                           n : Nat
                           M : Fin n.succ → Type v
                           M₁ : ι → Type v₁
                           M₂ : Type v₂
                           M₃ : Type v₃
                           M' : Type v'
                           inst✝⁹ : CommSemiring R
                           inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
                           inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
                           inst✝⁶ : AddCommMonoid M₂
                           inst✝⁵ : (i : Fin n.succ) → Module R (M i)
                           inst✝⁴ : (i : ι) → Module R (M₁ i)
                           inst✝³ : Module R M₂
                           f✝ f' : MultilinearMap R M₁ M₂
                           M₁' : ι → Type u_1
                           inst✝² : (i : ι) → AddCommMonoid (M₁' i)
                           inst✝¹ : (i : ι) → Module R (M₁' i)
                           α : Type u_2
                           f : MultilinearMap R M₁ M₂
                           s : Set ι
                           e : Equiv α ↑s
                           inst✝ : DecidablePred fun x => Membership.mem s x
                           ⊢ ∀ [inst : DecidableEq (Subtype fun a => Not (Membership.mem s a))] (m : (i : …
                         -/
  map_update_smul' := by intros; ext; simp
                                      /-
                                        🎉 no goals
                                      -/


open Classical in
/-- The `k`-th iterated derivative of a multilinear map `f` at the point `x`. It is a multilinear
map of `k` vectors `v₁, ..., vₖ` (with the same type as `x`), mapping them
to `∑ f (x₁, (v_{i₁})₂, x₃, ...)`, where at each index `j` one uses either `xⱼ` or one
of the `(vᵢ)ⱼ`, and each `vᵢ` has to be used exactly once.
The sum is parameterized by the embeddings of `Fin k` in the index type `ι` (or, equivalently,
by the subsets `s` of `ι` of cardinality `k` and then the bijections between `Fin k` and `s`).

For the continuous version, see `ContinuousMultilinearMap.iteratedFDeriv`. -/
protected noncomputable def iteratedFDeriv [Fintype ι]
    (f : MultilinearMap R M₁ M₂) (k : ℕ) (x : (i : ι) → M₁ i) :
    MultilinearMap R (fun (_ : Fin k) ↦ (∀ i, M₁ i)) M₂ :=
  ∑ e : Fin k ↪ ι, iteratedFDerivComponent f e.toEquivRange (fun i ↦ x i)


/-- If `f` is a collection of linear maps, then the construction `MultilinearMap.compLinearMap`
sending a multilinear map `g` to `g (f₁ ⬝ , ..., fₙ ⬝ )` is linear in `g`. -/
@[simps] def compLinearMapₗ (f : Π (i : ι), M₁ i →ₗ[R] M₁' i) :
    (MultilinearMap R M₁' M₂) →ₗ[R] MultilinearMap R M₁ M₂ where
  toFun := fun g ↦ g.compLinearMap f
  map_add' := fun _ _ ↦ rfl
  map_smul' := fun _ _ ↦ rfl


/-- If `f` is a collection of linear maps, then the construction `MultilinearMap.compLinearMap`
sending a multilinear map `g` to `g (f₁ ⬝ , ..., fₙ ⬝ )` is linear in `g` and multilinear in
`f₁, ..., fₙ`. -/
@[simps] def compLinearMapMultilinear :
  @MultilinearMap R ι (fun i ↦ M₁ i →ₗ[R] M₁' i)
    ((MultilinearMap R M₁' M₂) →ₗ[R] MultilinearMap R M₁ M₂) _ _ _
      (fun _ ↦ LinearMap.module) _ where
  toFun := MultilinearMap.compLinearMapₗ
  map_update_add' := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝¹ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝ : (i : ι) → Module R (M₁' i)
      ⊢ ∀ [inst : DecidableEq ι] (m : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' …
    -/
    intro _ f i f₁ f₂
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      f₁ f₂ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      ⊢ Eq (MultilinearMap.compLinearMapₗ (Function.update f i (HAdd.hAdd f₁ f₂))) ( …
    -/
    ext g x
    change (g fun j ↦ update f i (f₁ + f₂) j <| x j) =
        (g fun j ↦ update f i f₁ j <|x j) + g fun j ↦ update f i f₂ j (x j)
    /-
      case h.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      f₁ f₂ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      g : MultilinearMap R M₁' M₂
      x : (i : ι) → M₁ i
      ⊢ Eq (g fun j => (Function.update f i (HAdd.hAdd f₁ f₂) j) (x j)) (HAdd.hAdd ( …
    -/
    let c : Π (i : ι), (M₁ i →ₗ[R] M₁' i) → M₁' i := fun i f ↦ f (x i)
    /-
      case h.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      f₁ f₂ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      g : MultilinearMap R M₁' M₂
      x : (i : ι) → M₁ i
      c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
      ⊢ Eq (g fun j => (Function.update f i (HAdd.hAdd f₁ f₂) j) (x j)) (HAdd.hAdd ( …
    -/
    convert g.map_update_add (fun j ↦ f j (x j)) i (f₁ (x i)) (f₂ (x i)) with j j j
      /-
        case h.e'_2.h.e'_6.h
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁹ : CommSemiring R
        inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁶ : AddCommMonoid M₂
        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
        inst✝⁴ : (i : ι) → Module R (M₁ i)
        inst✝³ : Module R M₂
        f✝ f' : MultilinearMap R M₁ M₂
        M₁' : ι → Type u_1
        inst✝² : (i : ι) → AddCommMonoid (M₁' i)
        inst✝¹ : (i : ι) → Module R (M₁' i)
        inst✝ : DecidableEq ι
        f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        i : ι
        f₁ f₂ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        g : MultilinearMap R M₁' M₂
        x : (i : ι) → M₁ i
        c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
        j : ι
        ⊢ Eq ((Function.update f i (HAdd.hAdd f₁ f₂) j) (x j)) (Function.update (fun j …
      -/
    · exact Function.apply_update c f i (f₁ + f₂) j
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.e'_5.h.e'_6.h
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁹ : CommSemiring R
        inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁶ : AddCommMonoid M₂
        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
        inst✝⁴ : (i : ι) → Module R (M₁ i)
        inst✝³ : Module R M₂
        f✝ f' : MultilinearMap R M₁ M₂
        M₁' : ι → Type u_1
        inst✝² : (i : ι) → AddCommMonoid (M₁' i)
        inst✝¹ : (i : ι) → Module R (M₁' i)
        inst✝ : DecidableEq ι
        f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        i : ι
        f₁ f₂ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        g : MultilinearMap R M₁' M₂
        x : (i : ι) → M₁ i
        c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
        j : ι
        ⊢ Eq ((Function.update f i f₁ j) (x j)) (Function.update (fun j => (f j) (x j) …
      -/
    · exact Function.apply_update c f i f₁ j
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.e'_6.h.e'_6.h
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁹ : CommSemiring R
        inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁶ : AddCommMonoid M₂
        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
        inst✝⁴ : (i : ι) → Module R (M₁ i)
        inst✝³ : Module R M₂
        f✝ f' : MultilinearMap R M₁ M₂
        M₁' : ι → Type u_1
        inst✝² : (i : ι) → AddCommMonoid (M₁' i)
        inst✝¹ : (i : ι) → Module R (M₁' i)
        inst✝ : DecidableEq ι
        f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        i : ι
        f₁ f₂ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        g : MultilinearMap R M₁' M₂
        x : (i : ι) → M₁ i
        c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
        j : ι
        ⊢ Eq ((Function.update f i f₂ j) (x j)) (Function.update (fun j => (f j) (x j) …
      -/
    · exact Function.apply_update c f i f₂ j
      /-
        🎉 no goals
      -/
  map_update_smul' := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝¹ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝ : (i : ι) → Module R (M₁' i)
      ⊢ ∀ [inst : DecidableEq ι] (m : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' …
    -/
    intro _ f i a f₀
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      a : R
      f₀ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      ⊢ Eq (MultilinearMap.compLinearMapₗ (Function.update f i (HSMul.hSMul a f₀)))  …
    -/
    ext g x
    /-
      case h.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      a : R
      f₀ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      g : MultilinearMap R M₁' M₂
      x : (i : ι) → M₁ i
      ⊢ Eq (((MultilinearMap.compLinearMapₗ (Function.update f i (HSMul.hSMul a f₀)) …
    -/
    change (g fun j ↦ update f i (a • f₀) j <| x j) = a • g fun j ↦ update f i f₀ j (x j)
    /-
      case h.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      a : R
      f₀ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      g : MultilinearMap R M₁' M₂
      x : (i : ι) → M₁ i
      ⊢ Eq (g fun j => (Function.update f i (HSMul.hSMul a f₀) j) (x j)) (HSMul.hSMu …
    -/
    let c : Π (i : ι), (M₁ i →ₗ[R] M₁' i) → M₁' i := fun i f ↦ f (x i)
    /-
      case h.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁹ : CommSemiring R
      inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : (i : Fin n.succ) → Module R (M i)
      inst✝⁴ : (i : ι) → Module R (M₁ i)
      inst✝³ : Module R M₂
      f✝ f' : MultilinearMap R M₁ M₂
      M₁' : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹ : (i : ι) → Module R (M₁' i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      i : ι
      a : R
      f₀ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
      g : MultilinearMap R M₁' M₂
      x : (i : ι) → M₁ i
      c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
      ⊢ Eq (g fun j => (Function.update f i (HSMul.hSMul a f₀) j) (x j)) (HSMul.hSMu …
    -/
    convert g.map_update_smul (fun j ↦ f j (x j)) i a (f₀ (x i)) with j j j
      /-
        case h.e'_2.h.e'_6.h
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁹ : CommSemiring R
        inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁶ : AddCommMonoid M₂
        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
        inst✝⁴ : (i : ι) → Module R (M₁ i)
        inst✝³ : Module R M₂
        f✝ f' : MultilinearMap R M₁ M₂
        M₁' : ι → Type u_1
        inst✝² : (i : ι) → AddCommMonoid (M₁' i)
        inst✝¹ : (i : ι) → Module R (M₁' i)
        inst✝ : DecidableEq ι
        f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        i : ι
        a : R
        f₀ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        g : MultilinearMap R M₁' M₂
        x : (i : ι) → M₁ i
        c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
        j : ι
        ⊢ Eq ((Function.update f i (HSMul.hSMul a f₀) j) (x j)) (Function.update (fun  …
      -/
    · exact Function.apply_update c f i (a • f₀) j
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.e'_6.h.e'_6.h
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁹ : CommSemiring R
        inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁶ : AddCommMonoid M₂
        inst✝⁵ : (i : Fin n.succ) → Module R (M i)
        inst✝⁴ : (i : ι) → Module R (M₁ i)
        inst✝³ : Module R M₂
        f✝ f' : MultilinearMap R M₁ M₂
        M₁' : ι → Type u_1
        inst✝² : (i : ι) → AddCommMonoid (M₁' i)
        inst✝¹ : (i : ι) → Module R (M₁' i)
        inst✝ : DecidableEq ι
        f : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        i : ι
        a : R
        f₀ : LinearMap (RingHom.id R) (M₁ i) (M₁' i)
        g : MultilinearMap R M₁' M₂
        x : (i : ι) → M₁ i
        c : (i : ι) → LinearMap (RingHom.id R) (M₁ i) (M₁' i) → M₁' i := fun i f => f  …
        j : ι
        ⊢ Eq ((Function.update f i f₀ j) (x j)) (Function.update (fun j => (f j) (x j) …
      -/
    · exact Function.apply_update c f i f₀ j
      /-
        🎉 no goals
      -/


/--
Let `M₁ᵢ` and `M₁ᵢ'` be two families of `R`-modules and `M₂` an `R`-module.
Let us denote `Π i, M₁ᵢ` and `Π i, M₁ᵢ'` by `M` and `M'` respectively.
If `g` is a multilinear map `M' → M₂`, then `g` can be reinterpreted as a multilinear
map from `Π i, M₁ᵢ ⟶ M₁ᵢ'` to `M ⟶ M₂` via `(fᵢ) ↦ v ↦ g(fᵢ vᵢ)`.
-/
@[simps!] def piLinearMap :
    MultilinearMap R M₁' M₂ →ₗ[R]
    MultilinearMap R (fun i ↦ M₁ i →ₗ[R] M₁' i) (MultilinearMap R M₁ M₂) where
  toFun g := (LinearMap.applyₗ g).compMultilinearMap compLinearMapMultilinear
                 /-
                   R : Type uR
                   S : Type uS
                   ι : Type uι
                   n : Nat
                   M : Fin n.succ → Type v
                   M₁ : ι → Type v₁
                   M₂ : Type v₂
                   M₃ : Type v₃
                   M' : Type v'
                   inst✝⁸ : CommSemiring R
                   inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
                   inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
                   inst✝⁵ : AddCommMonoid M₂
                   inst✝⁴ : (i : Fin n.succ) → Module R (M i)
                   inst✝³ : (i : ι) → Module R (M₁ i)
                   inst✝² : Module R M₂
                   f f' : MultilinearMap R M₁ M₂
                   M₁' : ι → Type u_1
                   inst✝¹ : (i : ι) → AddCommMonoid (M₁' i)
                   inst✝ : (i : ι) → Module R (M₁' i)
                   ⊢ ∀ (x y : MultilinearMap R M₁' M₂), Eq ((fun g => (LinearMap.applyₗ g).compMu …
                 -/
  map_add' := by aesop
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type uR
                    S : Type uS
                    ι : Type uι
                    n : Nat
                    M : Fin n.succ → Type v
                    M₁ : ι → Type v₁
                    M₂ : Type v₂
                    M₃ : Type v₃
                    M' : Type v'
                    inst✝⁸ : CommSemiring R
                    inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
                    inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
                    inst✝⁵ : AddCommMonoid M₂
                    inst✝⁴ : (i : Fin n.succ) → Module R (M i)
                    inst✝³ : (i : ι) → Module R (M₁ i)
                    inst✝² : Module R M₂
                    f f' : MultilinearMap R M₁ M₂
                    M₁' : ι → Type u_1
                    inst✝¹ : (i : ι) → AddCommMonoid (M₁' i)
                    inst✝ : (i : ι) → Module R (M₁' i)
                    ⊢ ∀ (m : R) (x : MultilinearMap R M₁' M₂), Eq ({ toFun := fun g => (LinearMap. …
                  -/
  map_smul' := by aesop
                  /-
                    🎉 no goals
                  -/


/-- If one multiplies by `c i` the coordinates in a finset `s`, then the image under a multilinear
map is multiplied by `∏ i ∈ s, c i`. This is mainly an auxiliary statement to prove the result when
`s = univ`, given in `map_smul_univ`, although it can be useful in its own right as it does not
require the index set `ι` to be finite. -/
theorem map_piecewise_smul [DecidableEq ι] (c : ι → R) (m : ∀ i, M₁ i) (s : Finset ι) :
    f (s.piecewise (fun i => c i • m i) m) = (∏ i ∈ s, c i) • f m := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : CommSemiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    c : ι → R
    m : (i : ι) → M₁ i
    s : Finset ι
    ⊢ Eq (f (s.piecewise (fun i => HSMul.hSMul (c i) (m i)) m)) (HSMul.hSMul (s.pr …
  -/
  refine s.induction_on (by simp) ?_
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : CommSemiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    c : ι → R
    m : (i : ι) → M₁ i
    s : Finset ι
    ⊢ ∀ ⦃a : ι⦄ {s : Finset ι}, Not (Membership.mem s a) → Eq (f (s.piecewise (fun …
  -/
  intro j s j_not_mem_s Hrec
  have A :
    Function.update (s.piecewise (fun i => c i • m i) m) j (m j) =
      s.piecewise (fun i => c i • m i) m := by
    ext i
    by_cases h : i = j
    · rw [h]
      simp [j_not_mem_s]
    · simp [h]
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : CommSemiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    c : ι → R
    m : (i : ι) → M₁ i
    s✝ : Finset ι
    j : ι
    s : Finset ι
    j_not_mem_s : Not (Membership.mem s j)
    Hrec : Eq (f (s.piecewise (fun i => HSMul.hSMul (c i) (m i)) m)) (HSMul.hSMul  …
    A : Eq (Function.update (s.piecewise (fun i => HSMul.hSMul (c i) (m i)) m) j ( …
    ⊢ Eq (f ((Insert.insert j s).piecewise (fun i => HSMul.hSMul (c i) (m i)) m))  …
  -/
  rw [s.piecewise_insert, f.map_update_smul, A, Hrec]
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : CommSemiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    c : ι → R
    m : (i : ι) → M₁ i
    s✝ : Finset ι
    j : ι
    s : Finset ι
    j_not_mem_s : Not (Membership.mem s j)
    Hrec : Eq (f (s.piecewise (fun i => HSMul.hSMul (c i) (m i)) m)) (HSMul.hSMul  …
    A : Eq (Function.update (s.piecewise (fun i => HSMul.hSMul (c i) (m i)) m) j ( …
    ⊢ Eq (HSMul.hSMul (c j) (HSMul.hSMul (s.prod fun i => c i) (f m))) (HSMul.hSMu …
  -/
  simp [j_not_mem_s, mul_smul]
  /-
    🎉 no goals
  -/


/-- Multiplicativity of a multilinear map along all coordinates at the same time,
writing `f (fun i => c i • m i)` as `(∏ i, c i) • f m`. -/
theorem map_smul_univ [Fintype ι] (c : ι → R) (m : ∀ i, M₁ i) :
    (f fun i => c i • m i) = (∏ i, c i) • f m := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : CommSemiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝³ : AddCommMonoid M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : Fintype ι
    c : ι → R
    m : (i : ι) → M₁ i
    ⊢ Eq (f fun i => HSMul.hSMul (c i) (m i)) (HSMul.hSMul (Finset.univ.prod fun i …
  -/
  classical simpa using map_piecewise_smul f c m Finset.univ
  /-
    🎉 no goals
  -/


@[simp]
theorem map_update_smul_left [DecidableEq ι] [Fintype ι]
    (m : ∀ i, M₁ i) (i : ι) (c : R) (x : M₁ i) :
    f (update (c • m) i x) = c ^ (Fintype.card ι - 1) • f (update m i x) := by
  have :
    f ((Finset.univ.erase i).piecewise (c • update m i x) (update m i x)) =
      (∏ _i ∈ Finset.univ.erase i, c) • f (update m i x) :=
    map_piecewise_smul f _ _ _
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁶ : CommSemiring R
    inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : (i : ι) → Module R (M₁ i)
    inst✝² : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    m : (i : ι) → M₁ i
    i : ι
    c : R
    x : M₁ i
    this : Eq (f ((Finset.univ.erase i).piecewise (HSMul.hSMul c (Function.update  …
    ⊢ Eq (f (Function.update (HSMul.hSMul c m) i x)) (HSMul.hSMul (HPow.hPow c (HS …
  -/
  simpa [← Function.update_smul c m] using this
  /-
    🎉 no goals
  -/


/-- Given an `R`-algebra `A`, `mkPiAlgebra` is the multilinear map on `A^ι` associating
to `m` the product of all the `m i`.

See also `MultilinearMap.mkPiAlgebraFin` for a version that works with a non-commutative
algebra `A` but requires `ι = Fin n`. -/
protected def mkPiAlgebra : MultilinearMap R (fun _ : ι => A) A where
  toFun m := ∏ i, m i
                                /-
                                  R : Type uR
                                  S : Type uS
                                  ι : Type uι
                                  n : Nat
                                  M : Fin n.succ → Type v
                                  M₁ : ι → Type v₁
                                  M₂ : Type v₂
                                  M₃ : Type v₃
                                  M' : Type v'
                                  inst✝¹⁰ : CommSemiring R
                                  inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                                  inst✝⁸ : (i : Fin n.succ) → AddCommMonoid (M i)
                                  inst✝⁷ : AddCommMonoid M₂
                                  inst✝⁶ : (i : Fin n.succ) → Module R (M i)
                                  inst✝⁵ : (i : ι) → Module R (M₁ i)
                                  inst✝⁴ : Module R M₂
                                  f f' : MultilinearMap R M₁ M₂
                                  A : Type u_1
                                  inst✝³ : CommSemiring A
                                  inst✝² : Algebra R A
                                  inst✝¹ : Fintype ι
                                  inst✝ : DecidableEq ι
                                  m : ι → A
                                  i : ι
                                  x y : A
                                  ⊢ Eq ((fun m => Finset.univ.prod fun i => m i) (Function.update m i (HAdd.hAdd …
                                -/
  map_update_add' m i x y := by simp [Finset.prod_update_of_mem, add_mul]
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   R : Type uR
                                   S : Type uS
                                   ι : Type uι
                                   n : Nat
                                   M : Fin n.succ → Type v
                                   M₁ : ι → Type v₁
                                   M₂ : Type v₂
                                   M₃ : Type v₃
                                   M' : Type v'
                                   inst✝¹⁰ : CommSemiring R
                                   inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
                                   inst✝⁸ : (i : Fin n.succ) → AddCommMonoid (M i)
                                   inst✝⁷ : AddCommMonoid M₂
                                   inst✝⁶ : (i : Fin n.succ) → Module R (M i)
                                   inst✝⁵ : (i : ι) → Module R (M₁ i)
                                   inst✝⁴ : Module R M₂
                                   f f' : MultilinearMap R M₁ M₂
                                   A : Type u_1
                                   inst✝³ : CommSemiring A
                                   inst✝² : Algebra R A
                                   inst✝¹ : Fintype ι
                                   inst✝ : DecidableEq ι
                                   m : ι → A
                                   i : ι
                                   c : R
                                   x : A
                                   ⊢ Eq ((fun m => Finset.univ.prod fun i => m i) (Function.update m i (HSMul.hSM …
                                 -/
  map_update_smul' m i c x := by simp [Finset.prod_update_of_mem]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem mkPiAlgebra_apply (m : ι → A) : MultilinearMap.mkPiAlgebra R ι A m = ∏ i, m i :=
  rfl


/-- Given an `R`-algebra `A`, `mkPiAlgebraFin` is the multilinear map on `A^n` associating
to `m` the product of all the `m i`.

See also `MultilinearMap.mkPiAlgebra` for a version that assumes `[CommSemiring A]` but works
for `A^ι` with any finite type `ι`. -/
protected def mkPiAlgebraFin : MultilinearMap R (fun _ : Fin n => A) A where
  toFun m := (List.ofFn m).prod
  map_update_add' {dec} m i x y := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f f' : MultilinearMap R M₁ M₂
      A : Type u_1
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      dec : DecidableEq (Fin n)
      m : Fin n → A
      i : Fin n
      x y : A
      ⊢ Eq ((fun m => (List.ofFn m).prod) (Function.update m i (HAdd.hAdd x y))) (HA …
    -/
    rw [Subsingleton.elim dec (by infer_instance)]
    have : (List.finRange n).indexOf i < n := by
      simpa using List.indexOf_lt_length.2 (List.mem_finRange i)
    simp [List.ofFn_eq_map, (List.nodup_finRange n).map_update, List.prod_set, add_mul, this,
      mul_add, add_mul]
  map_update_smul' {dec} m i c x := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f f' : MultilinearMap R M₁ M₂
      A : Type u_1
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      dec : DecidableEq (Fin n)
      m : Fin n → A
      i : Fin n
      c : R
      x : A
      ⊢ Eq ((fun m => (List.ofFn m).prod) (Function.update m i (HSMul.hSMul c x))) ( …
    -/
    rw [Subsingleton.elim dec (by infer_instance)]
    have : (List.finRange n).indexOf i < n := by
      simpa using List.indexOf_lt_length.2 (List.mem_finRange i)
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M₂
      f f' : MultilinearMap R M₁ M₂
      A : Type u_1
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      dec : DecidableEq (Fin n)
      m : Fin n → A
      i : Fin n
      c : R
      x : A
      this : LT.lt (List.indexOf i (List.finRange n)) n
      ⊢ Eq ((fun m => (List.ofFn m).prod) (Function.update m i (HSMul.hSMul c x))) ( …
    -/
    simp [List.ofFn_eq_map, (List.nodup_finRange n).map_update, List.prod_set, this]
    /-
      🎉 no goals
    -/


@[simp]
theorem mkPiAlgebraFin_apply (m : Fin n → A) :
    MultilinearMap.mkPiAlgebraFin R n A m = (List.ofFn m).prod :=
  rfl


theorem mkPiAlgebraFin_apply_const (a : A) :
                                                                   /-
                                                                     R : Type uR
                                                                     n : Nat
                                                                     inst✝² : CommSemiring R
                                                                     A : Type u_1
                                                                     inst✝¹ : Semiring A
                                                                     inst✝ : Algebra R A
                                                                     a : A
                                                                     ⊢ Eq ((MultilinearMap.mkPiAlgebraFin R n A) fun x => a) (HPow.hPow a n)
                                                                   -/
    (MultilinearMap.mkPiAlgebraFin R n A fun _ => a) = a ^ n := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Given an `R`-multilinear map `f` taking values in `R`, `f.smulRight z` is the map
sending `m` to `f m • z`. -/
def smulRight (f : MultilinearMap R M₁ R) (z : M₂) : MultilinearMap R M₁ M₂ :=
  (LinearMap.smulRight LinearMap.id z).compMultilinearMap f


@[simp]
theorem smulRight_apply (f : MultilinearMap R M₁ R) (z : M₂) (m : ∀ i, M₁ i) :
    f.smulRight z m = f m • z :=
  rfl


/-- The canonical multilinear map on `R^ι` when `ι` is finite, associating to `m` the product of
all the `m i` (multiplied by a fixed reference element `z` in the target module). See also
`mkPiAlgebra` for a more general version. -/
protected def mkPiRing [Fintype ι] (z : M₂) : MultilinearMap R (fun _ : ι => R) M₂ :=
  (MultilinearMap.mkPiAlgebra R ι R).smulRight z


@[simp]
theorem mkPiRing_apply [Fintype ι] (z : M₂) (m : ι → R) :
    (MultilinearMap.mkPiRing R ι z : (ι → R) → M₂) m = (∏ i, m i) • z :=
  rfl


theorem mkPiRing_apply_one_eq_self [Fintype ι] (f : MultilinearMap R (fun _ : ι => R) M₂) :
    MultilinearMap.mkPiRing R ι (f fun _ => 1) = f := by
  /-
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    f : MultilinearMap R (fun x => R) M₂
    ⊢ Eq (MultilinearMap.mkPiRing R ι (f fun x => 1)) f
  -/
  ext m
  have : m = fun i => m i • (1 : R) := by
    ext j
    simp
  /-
    case H
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    f : MultilinearMap R (fun x => R) M₂
    m : ι → R
    this : Eq m fun i => HSMul.hSMul (m i) 1
    ⊢ Eq ((MultilinearMap.mkPiRing R ι (f fun x => 1)) m) (f m)
  -/
  conv_rhs => rw [this, f.map_smul_univ]
  /-
    case H
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    f : MultilinearMap R (fun x => R) M₂
    m : ι → R
    this : Eq m fun i => HSMul.hSMul (m i) 1
    ⊢ Eq ((MultilinearMap.mkPiRing R ι (f fun x => 1)) m) (HSMul.hSMul (Finset.uni …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mkPiRing_eq_iff [Fintype ι] {z₁ z₂ : M₂} :
    MultilinearMap.mkPiRing R ι z₁ = MultilinearMap.mkPiRing R ι z₂ ↔ z₁ = z₂ := by
  /-
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    z₁ z₂ : M₂
    ⊢ Iff (Eq (MultilinearMap.mkPiRing R ι z₁) (MultilinearMap.mkPiRing R ι z₂)) ( …
  -/
  simp_rw [MultilinearMap.ext_iff, mkPiRing_apply]
  /-
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    z₁ z₂ : M₂
    ⊢ Iff (∀ (x : ι → R), Eq (HSMul.hSMul (Finset.univ.prod fun i => x i) z₁) (HSM …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type uR
      ι : Type uι
      M₂ : Type v₂
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      z₁ z₂ : M₂
      h : ∀ (x : ι → R), Eq (HSMul.hSMul (Finset.univ.prod fun i => x i) z₁) (HSMul. …
      ⊢ Eq z₁ z₂
    -/
  · simpa using h fun _ => 1
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type uR
      ι : Type uι
      M₂ : Type v₂
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      z₁ z₂ : M₂
      h : Eq z₁ z₂
      ⊢ ∀ (x : ι → R), Eq (HSMul.hSMul (Finset.univ.prod fun i => x i) z₁) (HSMul.hS …
    -/
  · intro x
    /-
      case mpr
      R : Type uR
      ι : Type uι
      M₂ : Type v₂
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      z₁ z₂ : M₂
      h : Eq z₁ z₂
      x : ι → R
      ⊢ Eq (HSMul.hSMul (Finset.univ.prod fun i => x i) z₁) (HSMul.hSMul (Finset.uni …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem mkPiRing_zero [Fintype ι] : MultilinearMap.mkPiRing R ι (0 : M₂) = 0 := by
  /-
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    ⊢ Eq (MultilinearMap.mkPiRing R ι 0) 0
  -/
  ext; rw [mkPiRing_apply, smul_zero, MultilinearMap.zero_apply]
       /-
         🎉 no goals
       -/


theorem mkPiRing_eq_zero_iff [Fintype ι] (z : M₂) : MultilinearMap.mkPiRing R ι z = 0 ↔ z = 0 := by
  /-
    R : Type uR
    ι : Type uι
    M₂ : Type v₂
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Fintype ι
    z : M₂
    ⊢ Iff (Eq (MultilinearMap.mkPiRing R ι z) 0) (Eq z 0)
  -/
  rw [← mkPiRing_zero, mkPiRing_eq_iff]
  /-
    🎉 no goals
  -/


instance : Neg (MultilinearMap R M₁ M₂) :=
                                              /-
                                                R : Type uR
                                                S : Type uS
                                                ι : Type uι
                                                n : Nat
                                                M : Fin n.succ → Type v
                                                M₁ : ι → Type v₁
                                                M₂ : Type v₂
                                                M₃ : Type v₃
                                                M' : Type v'
                                                inst✝⁵ : Semiring R
                                                inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
                                                inst✝³ : AddCommGroup M₂
                                                inst✝² : (i : ι) → Module R (M₁ i)
                                                inst✝¹ : Module R M₂
                                                f✝ g f : MultilinearMap R M₁ M₂
                                                inst✝ : DecidableEq ι
                                                m : (i : ι) → M₁ i
                                                i : ι
                                                x y : M₁ i
                                                ⊢ Eq ((fun m => Neg.neg (f m)) (Function.update m i (HAdd.hAdd x y))) (HAdd.hA …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  ⟨fun f => ⟨fun m => -f m, fun m i x y => by simp [add_comm], fun m i c x => by simp⟩⟩
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
theorem neg_apply (m : ∀ i, M₁ i) : (-f) m = -f m :=
  rfl


instance : Sub (MultilinearMap R M₁ M₂) :=
  ⟨fun f g =>
    ⟨fun m => f m - g m, fun m i x y => by
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f✝ g✝ f g : MultilinearMap R M₁ M₂
        inst✝ : DecidableEq ι
        m : (i : ι) → M₁ i
        i : ι
        x y : M₁ i
        ⊢ Eq ((fun m => HSub.hSub (f m) (g m)) (Function.update m i (HAdd.hAdd x y)))  …
      -/
      simp only [MultilinearMap.map_update_add, sub_eq_add_neg, neg_add]
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f✝ g✝ f g : MultilinearMap R M₁ M₂
        inst✝ : DecidableEq ι
        m : (i : ι) → M₁ i
        i : ι
        x y : M₁ i
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (f (Function.update m i x)) (f (Function.update m i …
      -/
      /-
        🎉 no goals
      -/
      abel,
      /-
        🎉 no goals
      -/
                        /-
                          R : Type uR
                          S : Type uS
                          ι : Type uι
                          n : Nat
                          M : Fin n.succ → Type v
                          M₁ : ι → Type v₁
                          M₂ : Type v₂
                          M₃ : Type v₃
                          M' : Type v'
                          inst✝⁵ : Semiring R
                          inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
                          inst✝³ : AddCommGroup M₂
                          inst✝² : (i : ι) → Module R (M₁ i)
                          inst✝¹ : Module R M₂
                          f✝ g✝ f g : MultilinearMap R M₁ M₂
                          inst✝ : DecidableEq ι
                          m : (i : ι) → M₁ i
                          i : ι
                          c : R
                          x : M₁ i
                          ⊢ Eq ((fun m => HSub.hSub (f m) (g m)) (Function.update m i (HSMul.hSMul c x)) …
                        -/
      fun m i c x => by simp only [MultilinearMap.map_update_smul, smul_sub]⟩⟩
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem sub_apply (m : ∀ i, M₁ i) : (f - g) m = f m - g m :=
  rfl


instance : AddCommGroup (MultilinearMap R M₁ M₂) :=
  { MultilinearMap.addCommMonoid with
    neg_add_cancel := fun _ => MultilinearMap.ext fun _ => neg_add_cancel _
    sub_eq_add_neg := fun _ _ => MultilinearMap.ext fun _ => sub_eq_add_neg _ _
    zsmul := fun n f =>
      { toFun := fun m => n • f m
                                             /-
                                               R : Type uR
                                               S : Type uS
                                               ι : Type uι
                                               n✝ : Nat
                                               M : Fin n✝.succ → Type v
                                               M₁ : ι → Type v₁
                                               M₂ : Type v₂
                                               M₃ : Type v₃
                                               M' : Type v'
                                               inst✝⁵ : Semiring R
                                               inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
                                               inst✝³ : AddCommGroup M₂
                                               inst✝² : (i : ι) → Module R (M₁ i)
                                               inst✝¹ : Module R M₂
                                               f✝ g : MultilinearMap R M₁ M₂
                                               n : Int
                                               f : MultilinearMap R M₁ M₂
                                               inst✝ : DecidableEq ι
                                               m : (i : ι) → M₁ i
                                               i : ι
                                               x y : M₁ i
                                               ⊢ Eq ((fun m => HSMul.hSMul n (f m)) (Function.update m i (HAdd.hAdd x y))) (H …
                                             -/
        map_update_add' := fun m i x y => by simp [smul_add]
                                             /-
                                               🎉 no goals
                                             -/
                                              /-
                                                R : Type uR
                                                S : Type uS
                                                ι : Type uι
                                                n✝ : Nat
                                                M : Fin n✝.succ → Type v
                                                M₁ : ι → Type v₁
                                                M₂ : Type v₂
                                                M₃ : Type v₃
                                                M' : Type v'
                                                inst✝⁵ : Semiring R
                                                inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
                                                inst✝³ : AddCommGroup M₂
                                                inst✝² : (i : ι) → Module R (M₁ i)
                                                inst✝¹ : Module R M₂
                                                f✝ g : MultilinearMap R M₁ M₂
                                                n : Int
                                                f : MultilinearMap R M₁ M₂
                                                inst✝ : DecidableEq ι
                                                l : (i : ι) → M₁ i
                                                i : ι
                                                x : R
                                                d : M₁ i
                                                ⊢ Eq ((fun m => HSMul.hSMul n (f m)) (Function.update l i (HSMul.hSMul x d)))  …
                                              -/
        map_update_smul' := fun l i x d => by simp [← smul_comm x n (_ : M₂)] }
                                              /-
                                                🎉 no goals
                                              -/
    -- Porting note: changed from `AddCommGroup` to `SubNegMonoid`
    zsmul_zero' := fun _ => MultilinearMap.ext fun _ => SubNegMonoid.zsmul_zero' _
    zsmul_succ' := fun _ _ => MultilinearMap.ext fun _ => SubNegMonoid.zsmul_succ' _ _
    zsmul_neg' := fun _ _ => MultilinearMap.ext fun _ => SubNegMonoid.zsmul_neg' _ _ }


@[simp]
theorem map_update_neg [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (x : M₁ i) :
    f (update m i (-x)) = -f (update m i x) :=
  eq_neg_of_add_eq_zero_left <| by
    /-
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : DecidableEq ι
      m : (i : ι) → M₁ i
      i : ι
      x : M₁ i
      ⊢ Eq (HAdd.hAdd (f (Function.update m i (Neg.neg x))) (f (Function.update m i  …
    -/
    rw [← MultilinearMap.map_update_add, neg_add_cancel, f.map_coord_zero i (update_self i 0 m)]
    /-
      🎉 no goals
    -/



@[deprecated (since := "2024-11-03")] protected alias map_neg := MultilinearMap.map_update_neg


@[simp]
theorem map_update_sub [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (x y : M₁ i) :
    f (update m i (x - y)) = f (update m i x) - f (update m i y) := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    m : (i : ι) → M₁ i
    i : ι
    x y : M₁ i
    ⊢ Eq (f (Function.update m i (HSub.hSub x y))) (HSub.hSub (f (Function.update  …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, MultilinearMap.map_update_add, map_update_neg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-03")] protected alias map_sub := MultilinearMap.map_update_sub


lemma map_update [DecidableEq ι] (x : (i : ι) → M₁ i) (i : ι) (v : M₁ i)  :
    f (update x i v) = f x - f (update x i (x i - v)) := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : DecidableEq ι
    x : (i : ι) → M₁ i
    i : ι
    v : M₁ i
    ⊢ Eq (f (Function.update x i v)) (HSub.hSub (f x) (f (Function.update x i (HSu …
  -/
  rw [map_update_sub, update_eq_self, sub_sub_cancel]
  /-
    🎉 no goals
  -/


open Finset in
lemma map_sub_map_piecewise [LinearOrder ι] (a b : (i : ι) → M₁ i) (s : Finset ι) :
    f a - f (s.piecewise b a) =
    ∑ i ∈ s, f (fun j ↦ if j ∈ s → j < i then a j else if i = j then a j - b j else b j) := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : LinearOrder ι
    a b : (i : ι) → M₁ i
    s : Finset ι
    ⊢ Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite (M …
  -/
  refine s.induction_on_min ?_ fun k s hk ih ↦ ?_
    /-
      case refine_1
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b : (i : ι) → M₁ i
      s : Finset ι
      ⊢ Eq (HSub.hSub (f a) (f (EmptyCollection.emptyCollection.piecewise b a))) (Em …
    -/
  · rw [Finset.piecewise_empty, sum_empty, sub_self]
    /-
      🎉 no goals
    -/
  rw [Finset.piecewise_insert, map_update, ← sub_add, ih,
      add_comm, sum_insert (lt_irrefl _ <| hk k ·)]
  /-
    case refine_2
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : LinearOrder ι
    a b : (i : ι) → M₁ i
    s✝ : Finset ι
    k : ι
    s : Finset ι
    hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
    ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
    ⊢ Eq (HAdd.hAdd (f (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewis …
  -/
  simp_rw [s.mem_insert]
  /-
    case refine_2
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : LinearOrder ι
    a b : (i : ι) → M₁ i
    s✝ : Finset ι
    k : ι
    s : Finset ι
    hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
    ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
    ⊢ Eq (HAdd.hAdd (f (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewis …
  -/
  congr 1
    /-
      case refine_2.e_a
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b : (i : ι) → M₁ i
      s✝ : Finset ι
      k : ι
      s : Finset ι
      hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
      ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
      ⊢ Eq (f (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewise b a k) (b …
    -/
  · congr; ext i; split_ifs with h₁ h₂
      /-
        case pos
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b : (i : ι) → M₁ i
        s✝ : Finset ι
        k : ι
        s : Finset ι
        hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
        ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
        i : ι
        h₁ : Or (Eq i k) (Membership.mem s i) → LT.lt i k
        ⊢ Eq (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewise b a k) (b k) …
      -/
    · rw [update_of_ne, Finset.piecewise_eq_of_not_mem]
        /-
          case pos.hi
          R : Type uR
          ι : Type uι
          M₁ : ι → Type v₁
          M₂ : Type v₂
          inst✝⁵ : Semiring R
          inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
          inst✝³ : AddCommGroup M₂
          inst✝² : (i : ι) → Module R (M₁ i)
          inst✝¹ : Module R M₂
          f : MultilinearMap R M₁ M₂
          inst✝ : LinearOrder ι
          a b : (i : ι) → M₁ i
          s✝ : Finset ι
          k : ι
          s : Finset ι
          hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
          ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
          i : ι
          h₁ : Or (Eq i k) (Membership.mem s i) → LT.lt i k
          ⊢ Not (Membership.mem s i)
        -/
      · exact fun h ↦ (hk i h).not_lt (h₁ <| .inr h)
        /-
          🎉 no goals
        -/
        /-
          case pos.h
          R : Type uR
          ι : Type uι
          M₁ : ι → Type v₁
          M₂ : Type v₂
          inst✝⁵ : Semiring R
          inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
          inst✝³ : AddCommGroup M₂
          inst✝² : (i : ι) → Module R (M₁ i)
          inst✝¹ : Module R M₂
          f : MultilinearMap R M₁ M₂
          inst✝ : LinearOrder ι
          a b : (i : ι) → M₁ i
          s✝ : Finset ι
          k : ι
          s : Finset ι
          hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
          ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
          i : ι
          h₁ : Or (Eq i k) (Membership.mem s i) → LT.lt i k
          ⊢ Ne i k
        -/
      · exact fun h ↦ (h₁ <| .inl h).ne h
        /-
          🎉 no goals
        -/
      /-
        case pos
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b : (i : ι) → M₁ i
        s✝ : Finset ι
        k : ι
        s : Finset ι
        hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
        ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
        i : ι
        h₁ : Not (Or (Eq i k) (Membership.mem s i) → LT.lt i k)
        h₂ : Eq k i
        ⊢ Eq (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewise b a k) (b k) …
      -/
    · cases h₂
      /-
        case pos.refl
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b : (i : ι) → M₁ i
        s✝ : Finset ι
        k : ι
        s : Finset ι
        hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
        ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
        h₁ : Not (Or (Eq k k) (Membership.mem s k) → LT.lt k k)
        ⊢ Eq (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewise b a k) (b k) …
      -/
      rw [update_self, s.piecewise_eq_of_not_mem _ _ (lt_irrefl _ <| hk k ·)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b : (i : ι) → M₁ i
        s✝ : Finset ι
        k : ι
        s : Finset ι
        hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
        ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
        i : ι
        h₁ : Not (Or (Eq i k) (Membership.mem s i) → LT.lt i k)
        h₂ : Not (Eq k i)
        ⊢ Eq (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewise b a k) (b k) …
      -/
    · push_neg at h₁
      /-
        case neg
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b : (i : ι) → M₁ i
        s✝ : Finset ι
        k : ι
        s : Finset ι
        hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
        ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
        i : ι
        h₂ : Not (Eq k i)
        h₁ : And (Or (Eq i k) (Membership.mem s i)) (LE.le k i)
        ⊢ Eq (Function.update (s.piecewise b a) k (HSub.hSub (s.piecewise b a k) (b k) …
      -/
      rw [update_of_ne (Ne.symm h₂), s.piecewise_eq_of_mem _ _ (h₁.1.resolve_left <| Ne.symm h₂)]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.e_a
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b : (i : ι) → M₁ i
      s✝ : Finset ι
      k : ι
      s : Finset ι
      hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
      ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
      ⊢ Eq (s.sum fun i => f fun j => ite (Membership.mem s j → LT.lt j i) (a j) (it …
    -/
  · apply sum_congr rfl; intro i hi; congr; ext j; congr 1; apply propext
    /-
      case refine_2.e_a.h.e_6.h.h.e_c.a
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b : (i : ι) → M₁ i
      s✝ : Finset ι
      k : ι
      s : Finset ι
      hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
      ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
      i : ι
      hi : Membership.mem s i
      j : ι
      ⊢ Iff (Membership.mem s j → LT.lt j i) (Or (Eq j k) (Membership.mem s j) → LT. …
    -/
    simp_rw [imp_iff_not_or, not_or]; apply or_congr_left'
    /-
      case refine_2.e_a.h.e_6.h.h.e_c.a.h
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b : (i : ι) → M₁ i
      s✝ : Finset ι
      k : ι
      s : Finset ι
      hk : ∀ (x : ι), Membership.mem s x → LT.lt k x
      ih : Eq (HSub.hSub (f a) (f (s.piecewise b a))) (s.sum fun i => f fun j => ite …
      i : ι
      hi : Membership.mem s i
      j : ι
      ⊢ Not (LT.lt j i) → Iff (Not (Membership.mem s j)) (And (Not (Eq j k)) (Not (M …
    -/
    intro h; rw [and_iff_right]; rintro rfl; exact h (hk i hi)
                                             /-
                                               🎉 no goals
                                             -/


/-- This calculates the differences between the values of a multilinear map at
two arguments that differ on a finset `s` of `ι`. It requires a
linear order on `ι` in order to express the result. -/
lemma map_piecewise_sub_map_piecewise [LinearOrder ι] (a b v : (i : ι) → M₁ i) (s : Finset ι) :
    f (s.piecewise a v) - f (s.piecewise b v) = ∑ i ∈ s, f
      fun j ↦ if j ∈ s then if j < i then a j else if j = i then a j - b j else b j else v j := by
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : LinearOrder ι
    a b v : (i : ι) → M₁ i
    s : Finset ι
    ⊢ Eq (HSub.hSub (f (s.piecewise a v)) (f (s.piecewise b v))) (s.sum fun i => f …
  -/
  rw [← s.piecewise_idem_right b a, map_sub_map_piecewise]
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : LinearOrder ι
    a b v : (i : ι) → M₁ i
    s : Finset ι
    ⊢ Eq (s.sum fun i => f fun j => ite (Membership.mem s j → LT.lt j i) (s.piecew …
  -/
  refine Finset.sum_congr rfl fun i hi ↦ congr_arg f <| funext fun j ↦ ?_
  /-
    R : Type uR
    ι : Type uι
    M₁ : ι → Type v₁
    M₂ : Type v₂
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
    inst✝³ : AddCommGroup M₂
    inst✝² : (i : ι) → Module R (M₁ i)
    inst✝¹ : Module R M₂
    f : MultilinearMap R M₁ M₂
    inst✝ : LinearOrder ι
    a b v : (i : ι) → M₁ i
    s : Finset ι
    i : ι
    hi : Membership.mem s i
    j : ι
    ⊢ Eq (ite (Membership.mem s j → LT.lt j i) (s.piecewise a v j) (ite (Eq i j) ( …
  -/
  by_cases hjs : j ∈ s
    /-
      case pos
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b v : (i : ι) → M₁ i
      s : Finset ι
      i : ι
      hi : Membership.mem s i
      j : ι
      hjs : Membership.mem s j
      ⊢ Eq (ite (Membership.mem s j → LT.lt j i) (s.piecewise a v j) (ite (Eq i j) ( …
    -/
  · rw [if_pos hjs]; by_cases hji : j < i
      /-
        case pos
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b v : (i : ι) → M₁ i
        s : Finset ι
        i : ι
        hi : Membership.mem s i
        j : ι
        hjs : Membership.mem s j
        hji : LT.lt j i
        ⊢ Eq (ite (Membership.mem s j → LT.lt j i) (s.piecewise a v j) (ite (Eq i j) ( …
      -/
    · rw [if_pos fun _ ↦ hji, if_pos hji, s.piecewise_eq_of_mem _ _ hjs]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b v : (i : ι) → M₁ i
      s : Finset ι
      i : ι
      hi : Membership.mem s i
      j : ι
      hjs : Membership.mem s j
      hji : Not (LT.lt j i)
      ⊢ Eq (ite (Membership.mem s j → LT.lt j i) (s.piecewise a v j) (ite (Eq i j) ( …
    -/
    rw [if_neg (Classical.not_imp.mpr ⟨hjs, hji⟩), if_neg hji]
    /-
      case neg
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b v : (i : ι) → M₁ i
      s : Finset ι
      i : ι
      hi : Membership.mem s i
      j : ι
      hjs : Membership.mem s j
      hji : Not (LT.lt j i)
      ⊢ Eq (ite (Eq i j) (HSub.hSub (s.piecewise a v j) (b j)) (b j)) (ite (Eq j i)  …
    -/
    obtain rfl | hij := eq_or_ne i j
      /-
        case neg.inl
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b v : (i : ι) → M₁ i
        s : Finset ι
        i : ι
        hi hjs : Membership.mem s i
        hji : Not (LT.lt i i)
        ⊢ Eq (ite (Eq i i) (HSub.hSub (s.piecewise a v i) (b i)) (b i)) (ite (Eq i i)  …
      -/
    · rw [if_pos rfl, if_pos rfl, s.piecewise_eq_of_mem _ _ hi]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        R : Type uR
        ι : Type uι
        M₁ : ι → Type v₁
        M₂ : Type v₂
        inst✝⁵ : Semiring R
        inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
        inst✝³ : AddCommGroup M₂
        inst✝² : (i : ι) → Module R (M₁ i)
        inst✝¹ : Module R M₂
        f : MultilinearMap R M₁ M₂
        inst✝ : LinearOrder ι
        a b v : (i : ι) → M₁ i
        s : Finset ι
        i : ι
        hi : Membership.mem s i
        j : ι
        hjs : Membership.mem s j
        hji : Not (LT.lt j i)
        hij : Ne i j
        ⊢ Eq (ite (Eq i j) (HSub.hSub (s.piecewise a v j) (b j)) (b j)) (ite (Eq j i)  …
      -/
    · rw [if_neg hij, if_neg hij.symm]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type uR
      ι : Type uι
      M₁ : ι → Type v₁
      M₂ : Type v₂
      inst✝⁵ : Semiring R
      inst✝⁴ : (i : ι) → AddCommGroup (M₁ i)
      inst✝³ : AddCommGroup M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      f : MultilinearMap R M₁ M₂
      inst✝ : LinearOrder ι
      a b v : (i : ι) → M₁ i
      s : Finset ι
      i : ι
      hi : Membership.mem s i
      j : ι
      hjs : Not (Membership.mem s j)
      ⊢ Eq (ite (Membership.mem s j → LT.lt j i) (s.piecewise a v j) (ite (Eq i j) ( …
    -/
  · rw [if_neg hjs, if_pos fun h ↦ (hjs h).elim, s.piecewise_eq_of_not_mem _ _ hjs]
    /-
      🎉 no goals
    -/


open Finset in
lemma map_add_eq_map_add_linearDeriv_add [DecidableEq ι] [Fintype ι] (x h : (i : ι) → M₁ i) :
    f (x + h) = f x + f.linearDeriv x h + ∑ s with 2 ≤ #s, f (s.piecewise h x) := by
  rw [add_comm, map_add_univ, ← Finset.powerset_univ,
      ← sum_filter_add_sum_filter_not _ (2 ≤ #·)]
  simp_rw [not_le, Nat.lt_succ, le_iff_lt_or_eq (b := 1), Nat.lt_one_iff, filter_or,
    ← powersetCard_eq_filter, sum_union (univ.pairwise_disjoint_powersetCard zero_ne_one),
    powersetCard_zero, powersetCard_one, sum_singleton, Finset.piecewise_empty, sum_map,
    Function.Embedding.coeFn_mk, Finset.piecewise_singleton, linearDeriv_apply, add_comm]


open Finset in
/-- This expresses the difference between the values of a multilinear map
at two points "close to `x`" in terms of the "derivative" of the multilinear map at `x`
and of "second-order" terms. -/
lemma map_add_sub_map_add_sub_linearDeriv [DecidableEq ι] [Fintype ι] (x h h' : (i : ι) → M₁ i) :
    f (x + h) - f (x + h') - f.linearDeriv x (h - h') =
    ∑ s with 2 ≤ #s, (f (s.piecewise h x) - f (s.piecewise h' x)) := by
  simp_rw [map_add_eq_map_add_linearDeriv_add, add_assoc, add_sub_add_comm, sub_self, zero_add,
    ← LinearMap.map_sub, add_sub_cancel_left, sum_sub_distrib]


/-- When `ι` is finite, multilinear maps on `R^ι` with values in `M₂` are in bijection with `M₂`,
as such a multilinear map is completely determined by its value on the constant vector made of ones.
We register this bijection as a linear equivalence in `MultilinearMap.piRingEquiv`. -/
protected def piRingEquiv [Fintype ι] : M₂ ≃ₗ[R] MultilinearMap R (fun _ : ι => R) M₂ where
  toFun z := MultilinearMap.mkPiRing R ι z
  invFun f := f fun _ => 1
  map_add' z z' := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      z z' : M₂
      ⊢ Eq ((fun z => MultilinearMap.mkPiRing R ι z) (HAdd.hAdd z z')) (HAdd.hAdd (( …
    -/
    ext m
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      z z' : M₂
      m : ι → R
      ⊢ Eq (((fun z => MultilinearMap.mkPiRing R ι z) (HAdd.hAdd z z')) m) ((HAdd.hA …
    -/
    simp [smul_add]
    /-
      🎉 no goals
    -/
  map_smul' c z := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      c : R
      z : M₂
      ⊢ Eq ({ toFun := fun z => MultilinearMap.mkPiRing R ι z, map_add' := ⋯ }.toFun …
    -/
    ext m
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁵ : CommSemiring R
      inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : ι) → Module R (M₁ i)
      inst✝¹ : Module R M₂
      inst✝ : Fintype ι
      c : R
      z : M₂
      m : ι → R
      ⊢ Eq (({ toFun := fun z => MultilinearMap.mkPiRing R ι z, map_add' := ⋯ }.toFu …
    -/
    simp [smul_smul, mul_comm]
    /-
      🎉 no goals
    -/
                   /-
                     R : Type uR
                     S : Type uS
                     ι : Type uι
                     n : Nat
                     M : Fin n.succ → Type v
                     M₁ : ι → Type v₁
                     M₂ : Type v₂
                     M₃ : Type v₃
                     M' : Type v'
                     inst✝⁵ : CommSemiring R
                     inst✝⁴ : (i : ι) → AddCommMonoid (M₁ i)
                     inst✝³ : AddCommMonoid M₂
                     inst✝² : (i : ι) → Module R (M₁ i)
                     inst✝¹ : Module R M₂
                     inst✝ : Fintype ι
                     z : M₂
                     ⊢ Eq ((fun f => f fun x => 1) ({ toFun := fun z => MultilinearMap.mkPiRing R ι …
                   -/
  left_inv z := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv f := f.mkPiRing_apply_one_eq_self


/-- Given a linear map `f` from `M 0` to multilinear maps on `n` variables,
construct the corresponding multilinear map on `n+1` variables obtained by concatenating
the variables, given by `m ↦ f (m 0) (tail m)`-/
def LinearMap.uncurryLeft (f : M 0 →ₗ[R] MultilinearMap R (fun i : Fin n => M i.succ) M₂) :
    MultilinearMap R M M₂ where
  toFun m := f (m 0) (tail m)
  map_update_add' := @fun dec m i x y => by
    -- Porting note: `clear` not necessary in Lean 3 due to not being in the instance cache
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
      dec : DecidableEq (Fin n.succ)
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      x y : M i
      ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HAdd.hAdd x y))) …
    -/
    rw [Subsingleton.elim dec (by clear dec; infer_instance)]; clear dec
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      x y : M i
      ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HAdd.hAdd x y))) …
    -/
    by_cases h : i = 0
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : Eq i 0
        ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HAdd.hAdd x y))) …
      -/
    · subst i
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        x y : M 0
        ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m 0 (HAdd.hAdd x y))) …
      -/
      simp only [update_self, map_add, tail_update_zero, MultilinearMap.add_apply]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : Not (Eq i 0)
        ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HAdd.hAdd x y))) …
      -/
    · simp_rw [update_of_ne (Ne.symm h)]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : Not (Eq i 0)
        ⊢ Eq ((f (m 0)) (Fin.tail (Function.update m i (HAdd.hAdd x y)))) (HAdd.hAdd ( …
      -/
      revert x y
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : Not (Eq i 0)
        ⊢ ∀ (x y : M i), Eq ((f (m 0)) (Fin.tail (Function.update m i (HAdd.hAdd x y)) …
      -/
      rw [← succ_pred i h]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : Not (Eq i 0)
        ⊢ ∀ (x y : M (i.pred h).succ), Eq ((f (m 0)) (Fin.tail (Function.update m (i.p …
      -/
      intro x y
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : Not (Eq i 0)
        x y : M (i.pred h).succ
        ⊢ Eq ((f (m 0)) (Fin.tail (Function.update m (i.pred h).succ (HAdd.hAdd x y))) …
      -/
      rw [tail_update_succ, MultilinearMap.map_update_add, tail_update_succ, tail_update_succ]
      /-
        🎉 no goals
      -/
  map_update_smul' := @fun dec m i c x => by
    -- Porting note: `clear` not necessary in Lean 3 due to not being in the instance cache
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
      dec : DecidableEq (Fin n.succ)
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      c : R
      x : M i
      ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HSMul.hSMul c x) …
    -/
    rw [Subsingleton.elim dec (by clear dec; infer_instance)]; clear dec
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      c : R
      x : M i
      ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HSMul.hSMul c x) …
    -/
    by_cases h : i = 0
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : Eq i 0
        ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HSMul.hSMul c x) …
      -/
    · subst i
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        c : R
        x : M 0
        ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m 0 (HSMul.hSMul c x) …
      -/
      simp only [update_self, map_smul, tail_update_zero, MultilinearMap.smul_apply]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : Not (Eq i 0)
        ⊢ Eq ((fun m => (f (m 0)) (Fin.tail m)) (Function.update m i (HSMul.hSMul c x) …
      -/
    · simp_rw [update_of_ne (Ne.symm h)]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : Not (Eq i 0)
        ⊢ Eq ((f (m 0)) (Fin.tail (Function.update m i (HSMul.hSMul c x)))) (HSMul.hSM …
      -/
      revert x
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : Not (Eq i 0)
        ⊢ ∀ (x : M i), Eq ((f (m 0)) (Fin.tail (Function.update m i (HSMul.hSMul c x)) …
      -/
      rw [← succ_pred i h]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : Not (Eq i 0)
        ⊢ ∀ (x : M (i.pred h).succ), Eq ((f (m 0)) (Fin.tail (Function.update m (i.pre …
      -/
      intro x
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : Not (Eq i 0)
        x : M (i.pred h).succ
        ⊢ Eq ((f (m 0)) (Fin.tail (Function.update m (i.pred h).succ (HSMul.hSMul c x) …
      -/
      rw [tail_update_succ, tail_update_succ, MultilinearMap.map_update_smul]
      /-
        🎉 no goals
      -/


@[simp]
theorem LinearMap.uncurryLeft_apply (f : M 0 →ₗ[R] MultilinearMap R (fun i : Fin n => M i.succ) M₂)
    (m : ∀ i, M i) : f.uncurryLeft m = f (m 0) (tail m) :=
  rfl


/-- Given a multilinear map `f` in `n+1` variables, split the first variable to obtain
a linear map into multilinear maps in `n` variables, given by `x ↦ (m ↦ f (cons x m))`. -/
def MultilinearMap.curryLeft (f : MultilinearMap R M M₂) :
    M 0 →ₗ[R] MultilinearMap R (fun i : Fin n => M i.succ) M₂ where
  toFun x :=
    { toFun := fun m => f (cons x m)
      map_update_add' := @fun dec m i y y' => by
        -- Porting note: `clear` not necessary in Lean 3 due to not being in the instance cache
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁶ : CommSemiring R
          inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁴ : AddCommMonoid M'
          inst✝³ : AddCommMonoid M₂
          inst✝² : (i : Fin n.succ) → Module R (M i)
          inst✝¹ : Module R M'
          inst✝ : Module R M₂
          f : MultilinearMap R M M₂
          x : M 0
          dec : DecidableEq (Fin n)
          m : (i : Fin n) → M i.succ
          i : Fin n
          y y' : M i.succ
          ⊢ Eq ((fun m => f (Fin.cons x m)) (Function.update m i (HAdd.hAdd y y'))) (HAd …
        -/
        rw [Subsingleton.elim dec (by clear dec; infer_instance)]
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁶ : CommSemiring R
          inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁴ : AddCommMonoid M'
          inst✝³ : AddCommMonoid M₂
          inst✝² : (i : Fin n.succ) → Module R (M i)
          inst✝¹ : Module R M'
          inst✝ : Module R M₂
          f : MultilinearMap R M M₂
          x : M 0
          dec : DecidableEq (Fin n)
          m : (i : Fin n) → M i.succ
          i : Fin n
          y y' : M i.succ
          ⊢ Eq ((fun m => f (Fin.cons x m)) (Function.update m i (HAdd.hAdd y y'))) (HAd …
        -/
        simp
        /-
          🎉 no goals
        -/
      map_update_smul' := @fun dec m i y c => by
        -- Porting note: `clear` not necessary in Lean 3 due to not being in the instance cache
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁶ : CommSemiring R
          inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁴ : AddCommMonoid M'
          inst✝³ : AddCommMonoid M₂
          inst✝² : (i : Fin n.succ) → Module R (M i)
          inst✝¹ : Module R M'
          inst✝ : Module R M₂
          f : MultilinearMap R M M₂
          x : M 0
          dec : DecidableEq (Fin n)
          m : (i : Fin n) → M i.succ
          i : Fin n
          y : R
          c : M i.succ
          ⊢ Eq ((fun m => f (Fin.cons x m)) (Function.update m i (HSMul.hSMul y c))) (HS …
        -/
        rw [Subsingleton.elim dec (by clear dec; infer_instance)]
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁶ : CommSemiring R
          inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁴ : AddCommMonoid M'
          inst✝³ : AddCommMonoid M₂
          inst✝² : (i : Fin n.succ) → Module R (M i)
          inst✝¹ : Module R M'
          inst✝ : Module R M₂
          f : MultilinearMap R M M₂
          x : M 0
          dec : DecidableEq (Fin n)
          m : (i : Fin n) → M i.succ
          i : Fin n
          y : R
          c : M i.succ
          ⊢ Eq ((fun m => f (Fin.cons x m)) (Function.update m i (HSMul.hSMul y c))) (HS …
        -/
        simp }
        /-
          🎉 no goals
        -/
  map_add' x y := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      x y : M 0
      ⊢ Eq ((fun x => { toFun := fun m => f (Fin.cons x m), map_update_add' := ⋯, ma …
    -/
    ext m
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      x y : M 0
      m : (i : Fin n) → M i.succ
      ⊢ Eq (((fun x => { toFun := fun m => f (Fin.cons x m), map_update_add' := ⋯, m …
    -/
    exact cons_add f m x y
    /-
      🎉 no goals
    -/
  map_smul' c x := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      c : R
      x : M 0
      ⊢ Eq ({ toFun := fun x => { toFun := fun m => f (Fin.cons x m), map_update_add …
    -/
    ext m
    /-
      case H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      c : R
      x : M 0
      m : (i : Fin n) → M i.succ
      ⊢ Eq (({ toFun := fun x => { toFun := fun m => f (Fin.cons x m), map_update_ad …
    -/
    exact cons_smul f m c x
    /-
      🎉 no goals
    -/


@[simp]
theorem MultilinearMap.curryLeft_apply (f : MultilinearMap R M M₂) (x : M 0)
    (m : ∀ i : Fin n, M i.succ) : f.curryLeft x m = f (cons x m) :=
  rfl


@[simp]
theorem LinearMap.curry_uncurryLeft (f : M 0 →ₗ[R] MultilinearMap R (fun i :
    Fin n => M i.succ) M₂) : f.uncurryLeft.curryLeft = f := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
    ⊢ Eq f.uncurryLeft.curryLeft f
  -/
  ext m x
  /-
    case h.H
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
    m : M 0
    x : (i : Fin n) → M i.succ
    ⊢ Eq ((f.uncurryLeft.curryLeft m) x) ((f m) x)
  -/
  simp only [tail_cons, LinearMap.uncurryLeft_apply, MultilinearMap.curryLeft_apply]
  /-
    case h.H
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) (M 0) (MultilinearMap R (fun i => M i.succ) M₂)
    m : M 0
    x : (i : Fin n) → M i.succ
    ⊢ Eq ((f (Fin.cons m x 0)) x) ((f m) x)
  -/
  rw [cons_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem MultilinearMap.uncurry_curryLeft (f : MultilinearMap R M M₂) :
    f.curryLeft.uncurryLeft = f := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    ⊢ Eq f.curryLeft.uncurryLeft f
  -/
  ext m
  /-
    case H
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    m : (i : Fin n.succ) → M i
    ⊢ Eq (f.curryLeft.uncurryLeft m) (f m)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The space of multilinear maps on `Π (i : Fin (n+1)), M i` is canonically isomorphic to
the space of linear maps from `M 0` to the space of multilinear maps on
`Π (i : Fin n), M i.succ`, by separating the first variable. We register this isomorphism as a
linear isomorphism in `multilinearCurryLeftEquiv R M M₂`.

The direct and inverse maps are given by `f.curryLeft` and `f.uncurryLeft`. Use these
unless you need the full framework of linear equivs. -/
def multilinearCurryLeftEquiv :
    MultilinearMap R M M₂ ≃ₗ[R] (M 0 →ₗ[R] MultilinearMap R (fun i : Fin n => M i.succ) M₂) where
  toFun := MultilinearMap.curryLeft
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun := LinearMap.uncurryLeft
  left_inv := MultilinearMap.uncurry_curryLeft
  right_inv := LinearMap.curry_uncurryLeft


/-- Given a multilinear map `f` in `n` variables to the space of linear maps from `M (last n)` to
`M₂`, construct the corresponding multilinear map on `n+1` variables obtained by concatenating
the variables, given by `m ↦ f (init m) (m (last n))`-/
def MultilinearMap.uncurryRight
    (f : MultilinearMap R (fun i : Fin n => M (castSucc i)) (M (last n) →ₗ[R] M₂)) :
    MultilinearMap R M M₂ where
  toFun m := f (init m) (m (last n))
  map_update_add' {dec} m i x y := by
    -- Porting note: `clear` not necessary in Lean 3 due to not being in the instance cache
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
      dec : DecidableEq (Fin n.succ)
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      x y : M i
      ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HAdd. …
    -/
    rw [Subsingleton.elim dec (by clear dec; infer_instance)]; clear dec
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      x y : M i
      ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HAdd. …
    -/
    by_cases h : i.val < n
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : LT.lt (↑i) n
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HAdd. …
      -/
    · have : last n ≠ i := Ne.symm (ne_of_lt h)
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HAdd. …
      -/
      simp_rw [update_of_ne this]
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ Eq ((f (Fin.init (Function.update m i (HAdd.hAdd x y)))) (m (Fin.last n))) ( …
      -/
      revert x y
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ ∀ (x y : M i), Eq ((f (Fin.init (Function.update m i (HAdd.hAdd x y)))) (m ( …
      -/
      rw [(castSucc_castLT i h).symm]
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ ∀ (x y : M (i.castLT h).castSucc), Eq ((f (Fin.init (Function.update m (i.ca …
      -/
      intro x y
      rw [init_update_castSucc, MultilinearMap.map_update_add, init_update_castSucc,
        init_update_castSucc, LinearMap.add_apply]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        x y : M i
        h : Not (LT.lt (↑i) n)
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HAdd. …
      -/
    · revert x y
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : Not (LT.lt (↑i) n)
        ⊢ ∀ (x y : M i), Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.up …
      -/
      rw [eq_last_of_not_lt h]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : Not (LT.lt (↑i) n)
        ⊢ ∀ (x y : M (Fin.last n)), Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) ( …
      -/
      intro x y
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        h : Not (LT.lt (↑i) n)
        x y : M (Fin.last n)
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m (Fin.las …
      -/
      simp_rw [init_update_last, update_self, LinearMap.map_add]
      /-
        🎉 no goals
      -/
  map_update_smul' {dec} m i c x := by
    -- Porting note: `clear` not necessary in Lean 3 due to not being in the instance cache
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
      dec : DecidableEq (Fin n.succ)
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      c : R
      x : M i
      ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HSMul …
    -/
    rw [Subsingleton.elim dec (by clear dec; infer_instance)]; clear dec
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
      m : (i : Fin n.succ) → M i
      i : Fin n.succ
      c : R
      x : M i
      ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HSMul …
    -/
    by_cases h : i.val < n
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : LT.lt (↑i) n
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HSMul …
      -/
    · have : last n ≠ i := Ne.symm (ne_of_lt h)
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HSMul …
      -/
      simp_rw [update_of_ne this]
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ Eq ((f (Fin.init (Function.update m i (HSMul.hSMul c x)))) (m (Fin.last n))) …
      -/
      revert x
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ ∀ (x : M i), Eq ((f (Fin.init (Function.update m i (HSMul.hSMul c x)))) (m ( …
      -/
      rw [(castSucc_castLT i h).symm]
      /-
        case pos
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : LT.lt (↑i) n
        this : Ne (Fin.last n) i
        ⊢ ∀ (x : M (i.castLT h).castSucc), Eq ((f (Fin.init (Function.update m (i.cast …
      -/
      intro x
      rw [init_update_castSucc, init_update_castSucc, MultilinearMap.map_update_smul,
        LinearMap.smul_apply]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        x : M i
        h : Not (LT.lt (↑i) n)
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m i (HSMul …
      -/
    · revert x
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : Not (LT.lt (↑i) n)
        ⊢ ∀ (x : M i), Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.upda …
      -/
      rw [eq_last_of_not_lt h]
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : Not (LT.lt (↑i) n)
        ⊢ ∀ (x : M (Fin.last n)), Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Fu …
      -/
      intro x
      /-
        case neg
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁶ : CommSemiring R
        inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁴ : AddCommMonoid M'
        inst✝³ : AddCommMonoid M₂
        inst✝² : (i : Fin n.succ) → Module R (M i)
        inst✝¹ : Module R M'
        inst✝ : Module R M₂
        f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
        m : (i : Fin n.succ) → M i
        i : Fin n.succ
        c : R
        h : Not (LT.lt (↑i) n)
        x : M (Fin.last n)
        ⊢ Eq ((fun m => (f (Fin.init m)) (m (Fin.last n))) (Function.update m (Fin.las …
      -/
      simp_rw [update_self, init_update_last, map_smul]
      /-
        🎉 no goals
      -/


@[simp]
theorem MultilinearMap.uncurryRight_apply
    (f : MultilinearMap R (fun i : Fin n => M (castSucc i)) (M (last n) →ₗ[R] M₂))
    (m : ∀ i, M i) : f.uncurryRight m = f (init m) (m (last n)) :=
  rfl


/-- Given a multilinear map `f` in `n+1` variables, split the last variable to obtain
a multilinear map in `n` variables taking values in linear maps from `M (last n)` to `M₂`, given by
`m ↦ (x ↦ f (snoc m x))`. -/
def MultilinearMap.curryRight (f : MultilinearMap R M M₂) :
    MultilinearMap R (fun i : Fin n => M (Fin.castSucc i)) (M (last n) →ₗ[R] M₂) where
  toFun m :=
    { toFun := fun x => f (snoc m x)
                                /-
                                  R : Type uR
                                  S : Type uS
                                  ι : Type uι
                                  n : Nat
                                  M : Fin n.succ → Type v
                                  M₁ : ι → Type v₁
                                  M₂ : Type v₂
                                  M₃ : Type v₃
                                  M' : Type v'
                                  inst✝⁶ : CommSemiring R
                                  inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
                                  inst✝⁴ : AddCommMonoid M'
                                  inst✝³ : AddCommMonoid M₂
                                  inst✝² : (i : Fin n.succ) → Module R (M i)
                                  inst✝¹ : Module R M'
                                  inst✝ : Module R M₂
                                  f : MultilinearMap R M M₂
                                  m : (i : Fin n) → M i.castSucc
                                  x y : M (Fin.last n)
                                  ⊢ Eq ((fun x => f (Fin.snoc m x)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f (Fi …
                                -/
      map_add' := fun x y => by simp_rw [f.snoc_add]
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   R : Type uR
                                   S : Type uS
                                   ι : Type uι
                                   n : Nat
                                   M : Fin n.succ → Type v
                                   M₁ : ι → Type v₁
                                   M₂ : Type v₂
                                   M₃ : Type v₃
                                   M' : Type v'
                                   inst✝⁶ : CommSemiring R
                                   inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
                                   inst✝⁴ : AddCommMonoid M'
                                   inst✝³ : AddCommMonoid M₂
                                   inst✝² : (i : Fin n.succ) → Module R (M i)
                                   inst✝¹ : Module R M'
                                   inst✝ : Module R M₂
                                   f : MultilinearMap R M M₂
                                   m : (i : Fin n) → M i.castSucc
                                   c : R
                                   x : M (Fin.last n)
                                   ⊢ Eq ({ toFun := fun x => f (Fin.snoc m x), map_add' := ⋯ }.toFun (HSMul.hSMul …
                                 -/
      map_smul' := fun c x => by simp only [f.snoc_smul, RingHom.id_apply] }
                                 /-
                                   🎉 no goals
                                 -/
  map_update_add' := @fun dec m i x y => by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      dec : DecidableEq (Fin n)
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      x y : M i.castSucc
      ⊢ Eq ((fun m => { toFun := fun x => f (Fin.snoc m x), map_add' := ⋯, map_smul' …
    -/
    rw [Subsingleton.elim dec (by clear dec; infer_instance)]; clear dec
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      x y : M i.castSucc
      ⊢ Eq ((fun m => { toFun := fun x => f (Fin.snoc m x), map_add' := ⋯, map_smul' …
    -/
    ext z
    /-
      case h
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      x y : M i.castSucc
      z : M (Fin.last n)
      ⊢ Eq (((fun m => { toFun := fun x => f (Fin.snoc m x), map_add' := ⋯, map_smul …
    -/
    change f (snoc (update m i (x + y)) z) = f (snoc (update m i x) z) + f (snoc (update m i y) z)
    /-
      case h
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      x y : M i.castSucc
      z : M (Fin.last n)
      ⊢ Eq (f (Fin.snoc (Function.update m i (HAdd.hAdd x y)) z)) (HAdd.hAdd (f (Fin …
    -/
    rw [snoc_update, snoc_update, snoc_update, f.map_update_add]
    /-
      🎉 no goals
    -/
  map_update_smul' := @fun dec m i c x => by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      dec : DecidableEq (Fin n)
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      c : R
      x : M i.castSucc
      ⊢ Eq ((fun m => { toFun := fun x => f (Fin.snoc m x), map_add' := ⋯, map_smul' …
    -/
    rw [Subsingleton.elim dec (by clear dec; infer_instance)]; clear dec
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      c : R
      x : M i.castSucc
      ⊢ Eq ((fun m => { toFun := fun x => f (Fin.snoc m x), map_add' := ⋯, map_smul' …
    -/
    ext z
    /-
      case h
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      c : R
      x : M i.castSucc
      z : M (Fin.last n)
      ⊢ Eq (((fun m => { toFun := fun x => f (Fin.snoc m x), map_add' := ⋯, map_smul …
    -/
    change f (snoc (update m i (c • x)) z) = c • f (snoc (update m i x) z)
    /-
      case h
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      f : MultilinearMap R M M₂
      m : (i : Fin n) → M i.castSucc
      i : Fin n
      c : R
      x : M i.castSucc
      z : M (Fin.last n)
      ⊢ Eq (f (Fin.snoc (Function.update m i (HSMul.hSMul c x)) z)) (HSMul.hSMul c ( …
    -/
    rw [snoc_update, snoc_update, f.map_update_smul]
    /-
      🎉 no goals
    -/


@[simp]
theorem MultilinearMap.curryRight_apply (f : MultilinearMap R M M₂)
    (m : ∀ i : Fin n, M (castSucc i)) (x : M (last n)) : f.curryRight m x = f (snoc m x) :=
  rfl


@[simp]
theorem MultilinearMap.curry_uncurryRight
    (f : MultilinearMap R (fun i : Fin n => M (castSucc i)) (M (last n) →ₗ[R] M₂)) :
    f.uncurryRight.curryRight = f := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
    ⊢ Eq f.uncurryRight.curryRight f
  -/
  ext m x
  /-
    case H.h
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
    m : (i : Fin n) → M i.castSucc
    x : M (Fin.last n)
    ⊢ Eq ((f.uncurryRight.curryRight m) x) ((f m) x)
  -/
  simp only [snoc_last, MultilinearMap.curryRight_apply, MultilinearMap.uncurryRight_apply]
  /-
    case H.h
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R (fun i => M i.castSucc) (LinearMap (RingHom.id R) (M (Fin …
    m : (i : Fin n) → M i.castSucc
    x : M (Fin.last n)
    ⊢ Eq ((f (Fin.init (Fin.snoc m x))) x) ((f m) x)
  -/
  rw [init_snoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem MultilinearMap.uncurry_curryRight (f : MultilinearMap R M M₂) :
    f.curryRight.uncurryRight = f := by
  /-
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    ⊢ Eq f.curryRight.uncurryRight f
  -/
  ext m
  /-
    case H
    R : Type uR
    n : Nat
    M : Fin n.succ → Type v
    M₂ : Type v₂
    inst✝⁴ : CommSemiring R
    inst✝³ : (i : Fin n.succ) → AddCommMonoid (M i)
    inst✝² : AddCommMonoid M₂
    inst✝¹ : (i : Fin n.succ) → Module R (M i)
    inst✝ : Module R M₂
    f : MultilinearMap R M M₂
    m : (i : Fin n.succ) → M i
    ⊢ Eq (f.curryRight.uncurryRight m) (f m)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The space of multilinear maps on `Π (i : Fin (n+1)), M i` is canonically isomorphic to
the space of linear maps from the space of multilinear maps on `Π (i : Fin n), M (castSucc i)` to
the space of linear maps on `M (last n)`, by separating the last variable. We register this
isomorphism as a linear isomorphism in `multilinearCurryRightEquiv R M M₂`.

The direct and inverse maps are given by `f.curryRight` and `f.uncurryRight`. Use these
unless you need the full framework of linear equivs. -/
def multilinearCurryRightEquiv :
    MultilinearMap R M M₂ ≃ₗ[R]
      MultilinearMap R (fun i : Fin n => M (castSucc i)) (M (last n) →ₗ[R] M₂) where
  toFun := MultilinearMap.curryRight
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun := MultilinearMap.uncurryRight
  left_inv := MultilinearMap.uncurry_curryRight
  right_inv := MultilinearMap.curry_uncurryRight


/-- A multilinear map on `∀ i : ι ⊕ ι', M'` defines a multilinear map on `∀ i : ι, M'`
taking values in the space of multilinear maps on `∀ i : ι', M'`. -/
def currySum (f : MultilinearMap R (fun _ : ι ⊕ ι' => M') M₂) :
    MultilinearMap R (fun _ : ι => M') (MultilinearMap R (fun _ : ι' => M') M₂) where
  toFun u :=
    { toFun := fun v => f (Sum.elim u v)
      map_update_add' := fun v i x y => by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁷ : CommSemiring R
          inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁵ : AddCommMonoid M'
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : (i : Fin n.succ) → Module R (M i)
          inst✝² : Module R M'
          inst✝¹ : Module R M₂
          ι' : Type u_1
          f : MultilinearMap R (fun x => M') M₂
          u : ι → M'
          inst✝ : DecidableEq ι'
          v : ι' → M'
          i : ι'
          x y : M'
          ⊢ Eq ((fun v => f (Sum.elim u v)) (Function.update v i (HAdd.hAdd x y))) (HAdd …
        -/
        letI := Classical.decEq ι
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁷ : CommSemiring R
          inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁵ : AddCommMonoid M'
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : (i : Fin n.succ) → Module R (M i)
          inst✝² : Module R M'
          inst✝¹ : Module R M₂
          ι' : Type u_1
          f : MultilinearMap R (fun x => M') M₂
          u : ι → M'
          inst✝ : DecidableEq ι'
          v : ι' → M'
          i : ι'
          x y : M'
          this : DecidableEq ι := Classical.decEq ι
          ⊢ Eq ((fun v => f (Sum.elim u v)) (Function.update v i (HAdd.hAdd x y))) (HAdd …
        -/
        simp only [← Sum.update_elim_inr, f.map_update_add]
        /-
          🎉 no goals
        -/
      map_update_smul' := fun v i c x => by
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁷ : CommSemiring R
          inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁵ : AddCommMonoid M'
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : (i : Fin n.succ) → Module R (M i)
          inst✝² : Module R M'
          inst✝¹ : Module R M₂
          ι' : Type u_1
          f : MultilinearMap R (fun x => M') M₂
          u : ι → M'
          inst✝ : DecidableEq ι'
          v : ι' → M'
          i : ι'
          c : R
          x : M'
          ⊢ Eq ((fun v => f (Sum.elim u v)) (Function.update v i (HSMul.hSMul c x))) (HS …
        -/
        letI := Classical.decEq ι
        /-
          R : Type uR
          S : Type uS
          ι : Type uι
          n : Nat
          M : Fin n.succ → Type v
          M₁ : ι → Type v₁
          M₂ : Type v₂
          M₃ : Type v₃
          M' : Type v'
          inst✝⁷ : CommSemiring R
          inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
          inst✝⁵ : AddCommMonoid M'
          inst✝⁴ : AddCommMonoid M₂
          inst✝³ : (i : Fin n.succ) → Module R (M i)
          inst✝² : Module R M'
          inst✝¹ : Module R M₂
          ι' : Type u_1
          f : MultilinearMap R (fun x => M') M₂
          u : ι → M'
          inst✝ : DecidableEq ι'
          v : ι' → M'
          i : ι'
          c : R
          x : M'
          this : DecidableEq ι := Classical.decEq ι
          ⊢ Eq ((fun v => f (Sum.elim u v)) (Function.update v i (HSMul.hSMul c x))) (HS …
        -/
        simp only [← Sum.update_elim_inr, f.map_update_smul] }
        /-
          🎉 no goals
        -/
  map_update_add' u i x y :=
    ext fun v => by
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : Fin n.succ) → Module R (M i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        ι' : Type u_1
        f : MultilinearMap R (fun x => M') M₂
        inst✝ : DecidableEq ι
        u : ι → M'
        i : ι
        x y : M'
        v : ι' → M'
        ⊢ Eq (((fun u => { toFun := fun v => f (Sum.elim u v), map_update_add' := ⋯, m …
      -/
      letI := Classical.decEq ι'
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : Fin n.succ) → Module R (M i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        ι' : Type u_1
        f : MultilinearMap R (fun x => M') M₂
        inst✝ : DecidableEq ι
        u : ι → M'
        i : ι
        x y : M'
        v : ι' → M'
        this : DecidableEq ι' := Classical.decEq ι'
        ⊢ Eq (((fun u => { toFun := fun v => f (Sum.elim u v), map_update_add' := ⋯, m …
      -/
      simp only [MultilinearMap.coe_mk, add_apply, ← Sum.update_elim_inl, f.map_update_add]
      /-
        🎉 no goals
      -/
  map_update_smul' u i c x :=
    ext fun v => by
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : Fin n.succ) → Module R (M i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        ι' : Type u_1
        f : MultilinearMap R (fun x => M') M₂
        inst✝ : DecidableEq ι
        u : ι → M'
        i : ι
        c : R
        x : M'
        v : ι' → M'
        ⊢ Eq (((fun u => { toFun := fun v => f (Sum.elim u v), map_update_add' := ⋯, m …
      -/
      letI := Classical.decEq ι'
      /-
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : CommSemiring R
        inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : Fin n.succ) → Module R (M i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        ι' : Type u_1
        f : MultilinearMap R (fun x => M') M₂
        inst✝ : DecidableEq ι
        u : ι → M'
        i : ι
        c : R
        x : M'
        v : ι' → M'
        this : DecidableEq ι' := Classical.decEq ι'
        ⊢ Eq (((fun u => { toFun := fun v => f (Sum.elim u v), map_update_add' := ⋯, m …
      -/
      simp only [MultilinearMap.coe_mk, smul_apply, ← Sum.update_elim_inl, f.map_update_smul]
      /-
        🎉 no goals
      -/


@[simp]
theorem currySum_apply (f : MultilinearMap R (fun _ : ι ⊕ ι' => M') M₂) (u : ι → M')
    (v : ι' → M') : f.currySum u v = f (Sum.elim u v) :=
  rfl


/-- A multilinear map on `∀ i : ι, M'` taking values in the space of multilinear maps
on `∀ i : ι', M'` defines a multilinear map on `∀ i : ι ⊕ ι', M'`. -/
def uncurrySum (f : MultilinearMap R (fun _ : ι => M') (MultilinearMap R (fun _ : ι' => M') M₂)) :
    MultilinearMap R (fun _ : ι ⊕ ι' => M') M₂ where
  toFun u := f (u ∘ Sum.inl) (u ∘ Sum.inr)
  map_update_add' u i x y := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : Fin n.succ) → Module R (M i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      inst✝ : DecidableEq (Sum ι ι')
      u : Sum ι ι' → M'
      i : Sum ι ι'
      x y : M'
      ⊢ Eq ((fun u => (f (Function.comp u Sum.inl)) (Function.comp u Sum.inr)) (Func …
    -/
    letI := (@Sum.inl_injective ι ι').decidableEq
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : Fin n.succ) → Module R (M i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      inst✝ : DecidableEq (Sum ι ι')
      u : Sum ι ι' → M'
      i : Sum ι ι'
      x y : M'
      this : DecidableEq ι := ⋯.decidableEq
      ⊢ Eq ((fun u => (f (Function.comp u Sum.inl)) (Function.comp u Sum.inr)) (Func …
    -/
    letI := (@Sum.inr_injective ι ι').decidableEq
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : Fin n.succ) → Module R (M i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      inst✝ : DecidableEq (Sum ι ι')
      u : Sum ι ι' → M'
      i : Sum ι ι'
      x y : M'
      this✝ : DecidableEq ι := ⋯.decidableEq
      this : DecidableEq ι' := ⋯.decidableEq
      ⊢ Eq ((fun u => (f (Function.comp u Sum.inl)) (Function.comp u Sum.inr)) (Func …
    -/
    cases i <;>
      simp only [MultilinearMap.map_update_add, add_apply, Sum.update_inl_comp_inl,
        Sum.update_inl_comp_inr, Sum.update_inr_comp_inl, Sum.update_inr_comp_inr]
  map_update_smul' u i c x := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : Fin n.succ) → Module R (M i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      inst✝ : DecidableEq (Sum ι ι')
      u : Sum ι ι' → M'
      i : Sum ι ι'
      c : R
      x : M'
      ⊢ Eq ((fun u => (f (Function.comp u Sum.inl)) (Function.comp u Sum.inr)) (Func …
    -/
    letI := (@Sum.inl_injective ι ι').decidableEq
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : Fin n.succ) → Module R (M i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      inst✝ : DecidableEq (Sum ι ι')
      u : Sum ι ι' → M'
      i : Sum ι ι'
      c : R
      x : M'
      this : DecidableEq ι := ⋯.decidableEq
      ⊢ Eq ((fun u => (f (Function.comp u Sum.inl)) (Function.comp u Sum.inr)) (Func …
    -/
    letI := (@Sum.inr_injective ι ι').decidableEq
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : Fin n.succ) → Module R (M i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      inst✝ : DecidableEq (Sum ι ι')
      u : Sum ι ι' → M'
      i : Sum ι ι'
      c : R
      x : M'
      this✝ : DecidableEq ι := ⋯.decidableEq
      this : DecidableEq ι' := ⋯.decidableEq
      ⊢ Eq ((fun u => (f (Function.comp u Sum.inl)) (Function.comp u Sum.inr)) (Func …
    -/
    cases i <;>
      simp only [MultilinearMap.map_update_smul, smul_apply, Sum.update_inl_comp_inl,
        Sum.update_inl_comp_inr, Sum.update_inr_comp_inl, Sum.update_inr_comp_inr]


@[simp]
theorem uncurrySum_aux_apply
    (f : MultilinearMap R (fun _ : ι => M') (MultilinearMap R (fun _ : ι' => M') M₂))
    (u : ι ⊕ ι' → M') : f.uncurrySum u = f (u ∘ Sum.inl) (u ∘ Sum.inr) :=
  rfl


/-- Linear equivalence between the space of multilinear maps on `∀ i : ι ⊕ ι', M'` and the space
of multilinear maps on `∀ i : ι, M'` taking values in the space of multilinear maps
on `∀ i : ι', M'`. -/
def currySumEquiv :
    MultilinearMap R (fun _ : ι ⊕ ι' => M') M₂ ≃ₗ[R]
      MultilinearMap R (fun _ : ι => M') (MultilinearMap R (fun _ : ι' => M') M₂) where
  toFun := currySum
  invFun := uncurrySum
                                /-
                                  R : Type uR
                                  S : Type uS
                                  ι : Type uι
                                  n : Nat
                                  M : Fin n.succ → Type v
                                  M₁ : ι → Type v₁
                                  M₂ : Type v₂
                                  M₃ : Type v₃
                                  M' : Type v'
                                  inst✝⁶ : CommSemiring R
                                  inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
                                  inst✝⁴ : AddCommMonoid M'
                                  inst✝³ : AddCommMonoid M₂
                                  inst✝² : (i : Fin n.succ) → Module R (M i)
                                  inst✝¹ : Module R M'
                                  inst✝ : Module R M₂
                                  ι' : Type u_1
                                  f : MultilinearMap R (fun x => M') M₂
                                  u : Sum ι ι' → M'
                                  ⊢ Eq (({ toFun := MultilinearMap.currySum, map_add' := ⋯, map_smul' := ⋯ }.toF …
                                -/
  left_inv f := ext fun u => by simp
                                /-
                                  🎉 no goals
                                -/
  right_inv f := by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      ⊢ Eq ({ toFun := MultilinearMap.currySum, map_add' := ⋯, map_smul' := ⋯ }.toFu …
    -/
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      ι' : Type u_1
      f g : MultilinearMap R (fun x => M') M₂
      ⊢ Eq (HAdd.hAdd f g).currySum (HAdd.hAdd f.currySum g.currySum)
    -/
    ext
    /-
      case H.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      ι' : Type u_1
      f g : MultilinearMap R (fun x => M') M₂
      x✝¹ : ι → M'
      x✝ : ι' → M'
      ⊢ Eq (((HAdd.hAdd f g).currySum x✝¹) x✝) (((HAdd.hAdd f.currySum g.currySum) x …
    -/
    /-
      case H.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      ι' : Type u_1
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x✝¹ : ι → M'
      x✝ : ι' → M'
      ⊢ Eq ((({ toFun := MultilinearMap.currySum, map_add' := ⋯, map_smul' := ⋯ }.to …
    -/
    /-
      🎉 no goals
    -/
    simp
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      ι' : Type u_1
      c : R
      f : MultilinearMap R (fun x => M') M₂
      ⊢ Eq ({ toFun := MultilinearMap.currySum, map_add' := ⋯ }.toFun (HSMul.hSMul c …
    -/
    /-
      🎉 no goals
    -/
    /-
      case H.H
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁶ : CommSemiring R
      inst✝⁵ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : AddCommMonoid M₂
      inst✝² : (i : Fin n.succ) → Module R (M i)
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      ι' : Type u_1
      c : R
      f : MultilinearMap R (fun x => M') M₂
      x✝¹ : ι → M'
      x✝ : ι' → M'
      ⊢ Eq ((({ toFun := MultilinearMap.currySum, map_add' := ⋯ }.toFun (HSMul.hSMul …
    -/
  map_add' f g := by
    /-
      🎉 no goals
    -/
    ext
    rfl
  map_smul' c f := by
    ext
    rfl


@[simp]
theorem coe_currySumEquiv : ⇑(currySumEquiv R ι M₂ M' ι') = currySum :=
  rfl

-- Porting note: fixed missing letter `y` in name

@[simp]
theorem coe_currySumEquiv_symm : ⇑(currySumEquiv R ι M₂ M' ι').symm = uncurrySum :=
  rfl


/-- If `s : Finset (Fin n)` is a finite set of cardinality `k` and its complement has cardinality
`l`, then the space of multilinear maps on `fun i : Fin n => M'` is isomorphic to the space of
multilinear maps on `fun i : Fin k => M'` taking values in the space of multilinear maps
on `fun i : Fin l => M'`. -/
def curryFinFinset {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k) (hl : #sᶜ = l) :
    MultilinearMap R (fun _ : Fin n => M') M₂ ≃ₗ[R]
      MultilinearMap R (fun _ : Fin k => M') (MultilinearMap R (fun _ : Fin l => M') M₂) :=
  (domDomCongrLinearEquiv R R M' M₂ (finSumEquivOfFinset hk hl).symm).trans
    (currySumEquiv R (Fin k) M₂ M' (Fin l))


@[simp]
theorem curryFinFinset_apply {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k) (hl : #sᶜ = l)
    (f : MultilinearMap R (fun _ : Fin n => M') M₂) (mk : Fin k → M') (ml : Fin l → M') :
    curryFinFinset R M₂ M' hk hl f mk ml =
      f fun i => Sum.elim mk ml ((finSumEquivOfFinset hk hl).symm i) :=
  rfl


@[simp]
theorem curryFinFinset_symm_apply {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k)
    (hl : #sᶜ = l)
    (f : MultilinearMap R (fun _ : Fin k => M') (MultilinearMap R (fun _ : Fin l => M') M₂))
    (m : Fin n → M') :
    (curryFinFinset R M₂ M' hk hl).symm f m =
      f (fun i => m <| finSumEquivOfFinset hk hl (Sum.inl i)) fun i =>
        m <| finSumEquivOfFinset hk hl (Sum.inr i) :=
  rfl


theorem curryFinFinset_symm_apply_piecewise_const {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k)
    (hl : #sᶜ = l)
    (f : MultilinearMap R (fun _ : Fin k => M') (MultilinearMap R (fun _ : Fin l => M') M₂))
    (x y : M') :
    (curryFinFinset R M₂ M' hk hl).symm f (s.piecewise (fun _ => x) fun _ => y) =
      f (fun _ => x) fun _ => y := by
  /-
    R : Type uR
    M₂ : Type v₂
    M' : Type v'
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M'
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M'
    inst✝ : Module R M₂
    k l n : Nat
    s : Finset (Fin n)
    hk : Eq s.card k
    hl : Eq (HasCompl.compl s).card l
    f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
    x y : M'
    ⊢ Eq (((MultilinearMap.curryFinFinset R M₂ M' hk hl).symm f) (s.piecewise (fun …
  -/
  rw [curryFinFinset_symm_apply]; congr
    /-
      case h.e_5.h.h.e_6.h
      R : Type uR
      M₂ : Type v₂
      M' : Type v'
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M'
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      k l n : Nat
      s : Finset (Fin n)
      hk : Eq s.card k
      hl : Eq (HasCompl.compl s).card l
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x y : M'
      ⊢ Eq (fun i => s.piecewise (fun x_1 => x) (fun x => y) ((finSumEquivOfFinset h …
    -/
  · ext
    /-
      case h.e_5.h.h.e_6.h.h
      R : Type uR
      M₂ : Type v₂
      M' : Type v'
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M'
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      k l n : Nat
      s : Finset (Fin n)
      hk : Eq s.card k
      hl : Eq (HasCompl.compl s).card l
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x y : M'
      x✝ : Fin k
      ⊢ Eq (s.piecewise (fun x_1 => x) (fun x => y) ((finSumEquivOfFinset hk hl) (Su …
    -/
    rw [finSumEquivOfFinset_inl, Finset.piecewise_eq_of_mem]
    /-
      case h.e_5.h.h.e_6.h.h.hi
      R : Type uR
      M₂ : Type v₂
      M' : Type v'
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M'
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      k l n : Nat
      s : Finset (Fin n)
      hk : Eq s.card k
      hl : Eq (HasCompl.compl s).card l
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x y : M'
      x✝ : Fin k
      ⊢ Membership.mem s ((s.orderEmbOfFin hk) x✝)
    -/
    apply Finset.orderEmbOfFin_mem
    /-
      🎉 no goals
    -/
    /-
      case h.e_6.h
      R : Type uR
      M₂ : Type v₂
      M' : Type v'
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M'
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      k l n : Nat
      s : Finset (Fin n)
      hk : Eq s.card k
      hl : Eq (HasCompl.compl s).card l
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x y : M'
      ⊢ Eq (fun i => s.piecewise (fun x_1 => x) (fun x => y) ((finSumEquivOfFinset h …
    -/
  · ext
    /-
      case h.e_6.h.h
      R : Type uR
      M₂ : Type v₂
      M' : Type v'
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M'
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      k l n : Nat
      s : Finset (Fin n)
      hk : Eq s.card k
      hl : Eq (HasCompl.compl s).card l
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x y : M'
      x✝ : Fin l
      ⊢ Eq (s.piecewise (fun x_1 => x) (fun x => y) ((finSumEquivOfFinset hk hl) (Su …
    -/
    rw [finSumEquivOfFinset_inr, Finset.piecewise_eq_of_not_mem]
    /-
      case h.e_6.h.h.hi
      R : Type uR
      M₂ : Type v₂
      M' : Type v'
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M'
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R M'
      inst✝ : Module R M₂
      k l n : Nat
      s : Finset (Fin n)
      hk : Eq s.card k
      hl : Eq (HasCompl.compl s).card l
      f : MultilinearMap R (fun x => M') (MultilinearMap R (fun x => M') M₂)
      x y : M'
      x✝ : Fin l
      ⊢ Not (Membership.mem s (((HasCompl.compl s).orderEmbOfFin hl) x✝))
    -/
    exact Finset.mem_compl.1 (Finset.orderEmbOfFin_mem _ _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem curryFinFinset_symm_apply_const {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k)
    (hl : #sᶜ = l)
    (f : MultilinearMap R (fun _ : Fin k => M') (MultilinearMap R (fun _ : Fin l => M') M₂))
    (x : M') : ((curryFinFinset R M₂ M' hk hl).symm f fun _ => x) = f (fun _ => x) fun _ => x :=
  rfl


theorem curryFinFinset_apply_const {k l n : ℕ} {s : Finset (Fin n)} (hk : #s = k)
    (hl : #sᶜ = l) (f : MultilinearMap R (fun _ : Fin n => M') M₂) (x y : M') :
    (curryFinFinset R M₂ M' hk hl f (fun _ => x) fun _ => y) =
      f (s.piecewise (fun _ => x) fun _ => y) := by
  -- Porting note: `rw` fails
  /-
    R : Type uR
    M₂ : Type v₂
    M' : Type v'
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M'
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M'
    inst✝ : Module R M₂
    k l n : Nat
    s : Finset (Fin n)
    hk : Eq s.card k
    hl : Eq (HasCompl.compl s).card l
    f : MultilinearMap R (fun x => M') M₂
    x y : M'
    ⊢ Eq ((((MultilinearMap.curryFinFinset R M₂ M' hk hl) f) fun x_1 => x) fun x = …
  -/
  refine (curryFinFinset_symm_apply_piecewise_const hk hl _ _ _).symm.trans ?_
  /-
    R : Type uR
    M₂ : Type v₂
    M' : Type v'
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M'
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M'
    inst✝ : Module R M₂
    k l n : Nat
    s : Finset (Fin n)
    hk : Eq s.card k
    hl : Eq (HasCompl.compl s).card l
    f : MultilinearMap R (fun x => M') M₂
    x y : M'
    ⊢ Eq (((MultilinearMap.curryFinFinset R M₂ M' hk hl).symm ((MultilinearMap.cur …
  -/
  rw [LinearEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- The pushforward of an indexed collection of submodule `p i ⊆ M₁ i` by `f : M₁ → M₂`.

Note that this is not a submodule - it is not closed under addition. -/
def map [Nonempty ι] (f : MultilinearMap R M₁ M₂) (p : ∀ i, Submodule R (M₁ i)) :
    SubMulAction R M₂ where
  carrier := f '' { v | ∀ i, v i ∈ p i }
  smul_mem' := fun c _ ⟨x, hx, hf⟩ => by
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : Ring R
      inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      inst✝ : Nonempty ι
      f : MultilinearMap R M₁ M₂
      p : (i : ι) → Submodule R (M₁ i)
      c : R
      x✝¹ : M₂
      x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
      x : (i : ι) → M₁ i
      hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
      hf : Eq (f x) x✝¹
      ⊢ Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem (p  …
    -/
    let ⟨i⟩ := ‹Nonempty ι›
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : Ring R
      inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      inst✝ : Nonempty ι
      f : MultilinearMap R M₁ M₂
      p : (i : ι) → Submodule R (M₁ i)
      c : R
      x✝¹ : M₂
      x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
      x : (i : ι) → M₁ i
      hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
      hf : Eq (f x) x✝¹
      i : ι
      ⊢ Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem (p  …
    -/
    letI := Classical.decEq ι
    /-
      R : Type uR
      S : Type uS
      ι : Type uι
      n : Nat
      M : Fin n.succ → Type v
      M₁ : ι → Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      M' : Type v'
      inst✝⁷ : Ring R
      inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝⁵ : AddCommMonoid M'
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : (i : ι) → Module R (M₁ i)
      inst✝² : Module R M'
      inst✝¹ : Module R M₂
      inst✝ : Nonempty ι
      f : MultilinearMap R M₁ M₂
      p : (i : ι) → Submodule R (M₁ i)
      c : R
      x✝¹ : M₂
      x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
      x : (i : ι) → M₁ i
      hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
      hf : Eq (f x) x✝¹
      i : ι
      this : DecidableEq ι := Classical.decEq ι
      ⊢ Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem (p  …
    -/
    refine ⟨update x i (c • x i), fun j => if hij : j = i then ?_ else ?_, hf ▸ ?_⟩
      /-
        case refine_1
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : Ring R
        inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : ι) → Module R (M₁ i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        inst✝ : Nonempty ι
        f : MultilinearMap R M₁ M₂
        p : (i : ι) → Submodule R (M₁ i)
        c : R
        x✝¹ : M₂
        x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
        x : (i : ι) → M₁ i
        hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
        hf : Eq (f x) x✝¹
        i : ι
        this : DecidableEq ι := Classical.decEq ι
        j : ι
        hij : Eq j i
        ⊢ Membership.mem (p j) (Function.update x i (HSMul.hSMul c (x i)) j)
      -/
    · rw [hij, update_self]
      /-
        case refine_1
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : Ring R
        inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : ι) → Module R (M₁ i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        inst✝ : Nonempty ι
        f : MultilinearMap R M₁ M₂
        p : (i : ι) → Submodule R (M₁ i)
        c : R
        x✝¹ : M₂
        x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
        x : (i : ι) → M₁ i
        hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
        hf : Eq (f x) x✝¹
        i : ι
        this : DecidableEq ι := Classical.decEq ι
        j : ι
        hij : Eq j i
        ⊢ Membership.mem (p i) (HSMul.hSMul c (x i))
      -/
      exact (p i).smul_mem _ (hx i)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : Ring R
        inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : ι) → Module R (M₁ i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        inst✝ : Nonempty ι
        f : MultilinearMap R M₁ M₂
        p : (i : ι) → Submodule R (M₁ i)
        c : R
        x✝¹ : M₂
        x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
        x : (i : ι) → M₁ i
        hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
        hf : Eq (f x) x✝¹
        i : ι
        this : DecidableEq ι := Classical.decEq ι
        j : ι
        hij : Not (Eq j i)
        ⊢ Membership.mem (p j) (Function.update x i (HSMul.hSMul c (x i)) j)
      -/
    · rw [update_of_ne hij]
      /-
        case refine_2
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : Ring R
        inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : ι) → Module R (M₁ i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        inst✝ : Nonempty ι
        f : MultilinearMap R M₁ M₂
        p : (i : ι) → Submodule R (M₁ i)
        c : R
        x✝¹ : M₂
        x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
        x : (i : ι) → M₁ i
        hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
        hf : Eq (f x) x✝¹
        i : ι
        this : DecidableEq ι := Classical.decEq ι
        j : ι
        hij : Not (Eq j i)
        ⊢ Membership.mem (p j) (x j)
      -/
      exact hx j
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        R : Type uR
        S : Type uS
        ι : Type uι
        n : Nat
        M : Fin n.succ → Type v
        M₁ : ι → Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        M' : Type v'
        inst✝⁷ : Ring R
        inst✝⁶ : (i : ι) → AddCommMonoid (M₁ i)
        inst✝⁵ : AddCommMonoid M'
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : (i : ι) → Module R (M₁ i)
        inst✝² : Module R M'
        inst✝¹ : Module R M₂
        inst✝ : Nonempty ι
        f : MultilinearMap R M₁ M₂
        p : (i : ι) → Submodule R (M₁ i)
        c : R
        x✝¹ : M₂
        x✝ : Membership.mem (Set.image (⇑f) (setOf fun v => ∀ (i : ι), Membership.mem  …
        x : (i : ι) → M₁ i
        hx : Membership.mem (setOf fun v => ∀ (i : ι), Membership.mem (p i) (v i)) x
        hf : Eq (f x) x✝¹
        i : ι
        this : DecidableEq ι := Classical.decEq ι
        ⊢ Eq (f (Function.update x i (HSMul.hSMul c (x i)))) (HSMul.hSMul c (f x))
      -/
    · rw [f.map_update_smul, update_eq_self]
      /-
        🎉 no goals
      -/


/-- The map is always nonempty. This lemma is needed to apply `SubMulAction.zero_mem`. -/
theorem map_nonempty [Nonempty ι] (f : MultilinearMap R M₁ M₂) (p : ∀ i, Submodule R (M₁ i)) :
    (map f p : Set M₂).Nonempty :=
  ⟨f 0, 0, fun i => (p i).zero_mem, rfl⟩


/-- The range of a multilinear map, closed under scalar multiplication. -/
def range [Nonempty ι] (f : MultilinearMap R M₁ M₂) : SubMulAction R M₂ :=
  f.map fun _ => ⊤


