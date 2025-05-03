/-- A module `M` is Hausdorff with respect to an ideal `I` if `⋂ I^n M = 0`. -/
class IsHausdorff : Prop where
  haus' : ∀ x : M, (∀ n : ℕ, x ≡ 0 [SMOD (I ^ n • ⊤ : Submodule R M)]) → x = 0


/-- A module `M` is precomplete with respect to an ideal `I` if every Cauchy sequence converges. -/
class IsPrecomplete : Prop where
  prec' : ∀ f : ℕ → M, (∀ {m n}, m ≤ n → f m ≡ f n [SMOD (I ^ m • ⊤ : Submodule R M)]) →
    ∃ L : M, ∀ n, f n ≡ L [SMOD (I ^ n • ⊤ : Submodule R M)]


/-- A module `M` is `I`-adically complete if it is Hausdorff and precomplete. -/
class IsAdicComplete extends IsHausdorff I M, IsPrecomplete I M : Prop


theorem IsHausdorff.haus (_ : IsHausdorff I M) :
    ∀ x : M, (∀ n : ℕ, x ≡ 0 [SMOD (I ^ n • ⊤ : Submodule R M)]) → x = 0 :=
  IsHausdorff.haus'


theorem isHausdorff_iff :
    IsHausdorff I M ↔ ∀ x : M, (∀ n : ℕ, x ≡ 0 [SMOD (I ^ n • ⊤ : Submodule R M)]) → x = 0 :=
  ⟨IsHausdorff.haus, fun h => ⟨h⟩⟩


theorem IsPrecomplete.prec (_ : IsPrecomplete I M) {f : ℕ → M} :
    (∀ {m n}, m ≤ n → f m ≡ f n [SMOD (I ^ m • ⊤ : Submodule R M)]) →
      ∃ L : M, ∀ n, f n ≡ L [SMOD (I ^ n • ⊤ : Submodule R M)] :=
  IsPrecomplete.prec' _


theorem isPrecomplete_iff :
    IsPrecomplete I M ↔
      ∀ f : ℕ → M,
        (∀ {m n}, m ≤ n → f m ≡ f n [SMOD (I ^ m • ⊤ : Submodule R M)]) →
          ∃ L : M, ∀ n, f n ≡ L [SMOD (I ^ n • ⊤ : Submodule R M)] :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


/-- The Hausdorffification of a module with respect to an ideal. -/
abbrev Hausdorffification : Type _ :=
  M ⧸ (⨅ n : ℕ, I ^ n • ⊤ : Submodule R M)


/-- The canonical linear map `M ⧸ (I ^ n • ⊤) →ₗ[R] M ⧸ (I ^ m • ⊤)` for `m ≤ n` used
to define `AdicCompletion`. -/
def AdicCompletion.transitionMap {m n : ℕ} (hmn : m ≤ n) :
    M ⧸ (I ^ n • ⊤ : Submodule R M) →ₗ[R] M ⧸ (I ^ m • ⊤ : Submodule R M) :=
  liftQ (I ^ n • ⊤ : Submodule R M) (mkQ (I ^ m • ⊤ : Submodule R M)) (by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le (HSMul.hSMul (HPow.hPow I n) Top.top) (LinearMap.ker (HSMul.hSMul (HPo …
    -/
    rw [ker_mkQ]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le (HSMul.hSMul (HPow.hPow I n) Top.top) (HSMul.hSMul (HPow.hPow I m) Top …
    -/
    exact smul_mono (Ideal.pow_le_pow_right hmn) le_rfl)
    /-
      🎉 no goals
    -/


/-- The completion of a module with respect to an ideal. This is not necessarily Hausdorff.
In fact, this is only complete if the ideal is finitely generated. -/
def AdicCompletion : Type _ :=
  { f : ∀ n : ℕ, M ⧸ (I ^ n • ⊤ : Submodule R M) //
    ∀ {m n} (hmn : m ≤ n), AdicCompletion.transitionMap I M hmn (f n) = f m }


instance bot : IsHausdorff (⊥ : Ideal R) M :=
                  /-
                    R : Type u_1
                    S : Type u_2
                    T : Type u_3
                    inst✝⁴ : CommRing R
                    I : Ideal R
                    M : Type u_4
                    inst✝³ : AddCommGroup M
                    inst✝² : Module R M
                    N : Type u_5
                    inst✝¹ : AddCommGroup N
                    inst✝ : Module R N
                    x : M
                    hx : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow Bot.bot n) Top.top) x 0
                    ⊢ Eq x 0
                  -/
  ⟨fun x hx => by simpa only [pow_one ⊥, bot_smul, SModEq.bot] using hx 1⟩
                  /-
                    🎉 no goals
                  -/


protected theorem subsingleton (h : IsHausdorff (⊤ : Ideal R) M) : Subsingleton M :=
  ⟨fun x y => eq_of_sub_eq_zero <| h.haus (x - y) fun n => by
    /-
      R : Type u_1
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      h : IsHausdorff Top.top M
      x y : M
      n : Nat
      ⊢ SModEq (HSMul.hSMul (HPow.hPow Top.top n) Top.top) (HSub.hSub x y) 0
    -/
    rw [Ideal.top_pow, top_smul]
    /-
      R : Type u_1
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      h : IsHausdorff Top.top M
      x y : M
      n : Nat
      ⊢ SModEq Top.top (HSub.hSub x y) 0
    -/
    exact SModEq.top⟩
    /-
      🎉 no goals
    -/


instance (priority := 100) of_subsingleton [Subsingleton M] : IsHausdorff I M :=
  ⟨fun _ _ => Subsingleton.elim _ _⟩


theorem iInf_pow_smul (h : IsHausdorff I M) : (⨅ n : ℕ, I ^ n • ⊤ : Submodule R M) = ⊥ :=
  eq_bot_iff.2 fun x hx =>
    (mem_bot _).2 <| h.haus x fun n => SModEq.zero.2 <| (mem_iInf fun n : ℕ => I ^ n • ⊤).1 hx n


/-- The canonical linear map to the Hausdorffification. -/
def of : M →ₗ[R] Hausdorffification I M :=
  mkQ _


@[elab_as_elim]
theorem induction_on {C : Hausdorffification I M → Prop} (x : Hausdorffification I M)
    (ih : ∀ x, C (of I M x)) : C x :=
  Quotient.inductionOn' x ih


instance : IsHausdorff I (Hausdorffification I M) :=
  ⟨fun x => Quotient.inductionOn' x fun x hx =>
    (Quotient.mk_eq_zero _).2 <| (mem_iInf _).2 fun n => by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        x✝ : Hausdorffification I M
        x : M
        hx : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (Quotient.mk''  …
        n : Nat
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) x
      -/
      have := comap_map_mkQ (⨅ n : ℕ, I ^ n • ⊤ : Submodule R M) (I ^ n • ⊤)
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        x✝ : Hausdorffification I M
        x : M
        hx : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (Quotient.mk''  …
        n : Nat
        this : Eq (Submodule.comap (iInf fun n => HSMul.hSMul (HPow.hPow I n) Top.top) …
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) x
      -/
      simp only [sup_of_le_right (iInf_le (fun n => (I ^ n • ⊤ : Submodule R M)) n)] at this
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        x✝ : Hausdorffification I M
        x : M
        hx : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (Quotient.mk''  …
        n : Nat
        this : Eq (Submodule.comap (iInf fun n => HSMul.hSMul (HPow.hPow I n) Top.top) …
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) x
      -/
      rw [← this, map_smul'', mem_comap, Submodule.map_top, range_mkQ, ← SModEq.zero]
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        x✝ : Hausdorffification I M
        x : M
        hx : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (Quotient.mk''  …
        n : Nat
        this : Eq (Submodule.comap (iInf fun n => HSMul.hSMul (HPow.hPow I n) Top.top) …
        ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) ((iInf fun n => HSMul.hSMul (HP …
      -/
      exact hx n⟩
      /-
        🎉 no goals
      -/


/-- Universal property of Hausdorffification: any linear map to a Hausdorff module extends to a
unique map from the Hausdorffification. -/
def lift (f : M →ₗ[R] N) : Hausdorffification I M →ₗ[R] N :=
  liftQ _ f <| map_le_iff_le_comap.1 <| h.iInf_pow_smul ▸ le_iInf fun n =>
    le_trans (map_mono <| iInf_le _ n) <| by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        h : IsHausdorff I N
        f : LinearMap (RingHom.id R) M N
        n : Nat
        ⊢ LE.le (Submodule.map f (HSMul.hSMul (HPow.hPow I n) Top.top)) (HSMul.hSMul ( …
      -/
      rw [map_smul'']
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        h : IsHausdorff I N
        f : LinearMap (RingHom.id R) M N
        n : Nat
        ⊢ LE.le (HSMul.hSMul (HPow.hPow I n) (Submodule.map f Top.top)) (HSMul.hSMul ( …
      -/
      exact smul_mono le_rfl le_top
      /-
        🎉 no goals
      -/


theorem lift_of (f : M →ₗ[R] N) (x : M) : lift I f (of I M x) = f x :=
  rfl


theorem lift_comp_of (f : M →ₗ[R] N) : (lift I f).comp (of I M) = f :=
  LinearMap.ext fun _ => rfl


/-- Uniqueness of lift. -/
theorem lift_eq (f : M →ₗ[R] N) (g : Hausdorffification I M →ₗ[R] N) (hg : g.comp (of I M) = f) :
    g = lift I f :=
                                                    /-
                                                      R : Type u_1
                                                      inst✝⁴ : CommRing R
                                                      I : Ideal R
                                                      M : Type u_4
                                                      inst✝³ : AddCommGroup M
                                                      inst✝² : Module R M
                                                      N : Type u_5
                                                      inst✝¹ : AddCommGroup N
                                                      inst✝ : Module R N
                                                      h : IsHausdorff I N
                                                      f : LinearMap (RingHom.id R) M N
                                                      g : LinearMap (RingHom.id R) (Hausdorffification I M) N
                                                      hg : Eq (g.comp (Hausdorffification.of I M)) f
                                                      x✝ : Hausdorffification I M
                                                      x : M
                                                      ⊢ Eq (g ((Hausdorffification.of I M) x)) ((Hausdorffification.lift I f) ((Haus …
                                                    -/
  LinearMap.ext fun x => induction_on x fun x => by rw [lift_of, ← hg, LinearMap.comp_apply]
                                                    /-
                                                      🎉 no goals
                                                    -/


instance bot : IsPrecomplete (⊥ : Ideal R) M := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ IsPrecomplete Bot.bot M
  -/
  refine ⟨fun f hf => ⟨f 1, fun n => ?_⟩⟩
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : Nat → M
    hf : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow Bot.bot m) Top. …
    n : Nat
    ⊢ SModEq (HSMul.hSMul (HPow.hPow Bot.bot n) Top.top) (f n) (f 1)
  -/
  cases' n with n
    /-
      case zero
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : Nat → M
      hf : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow Bot.bot m) Top. …
      ⊢ SModEq (HSMul.hSMul (HPow.hPow Bot.bot 0) Top.top) (f 0) (f 1)
    -/
  · rw [pow_zero, Ideal.one_eq_top, top_smul]
    /-
      case zero
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : Nat → M
      hf : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow Bot.bot m) Top. …
      ⊢ SModEq Top.top (f 0) (f 1)
    -/
    exact SModEq.top
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : Nat → M
    hf : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow Bot.bot m) Top. …
    n : Nat
    ⊢ SModEq (HSMul.hSMul (HPow.hPow Bot.bot (HAdd.hAdd n 1)) Top.top) (f (HAdd.hA …
  -/
  specialize hf (Nat.le_add_left 1 n)
  /-
    case succ
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : Nat → M
    n : Nat
    hf : SModEq (HSMul.hSMul (HPow.hPow Bot.bot 1) Top.top) (f 1) (f (HAdd.hAdd n  …
    ⊢ SModEq (HSMul.hSMul (HPow.hPow Bot.bot (HAdd.hAdd n 1)) Top.top) (f (HAdd.hA …
  -/
  rw [pow_one, bot_smul, SModEq.bot] at hf; rw [hf]
                                            /-
                                              🎉 no goals
                                            -/


instance top : IsPrecomplete (⊤ : Ideal R) M :=
  ⟨fun f _ =>
    ⟨0, fun n => by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : Nat → M
        x✝ : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow Top.top m) Top. …
        n : Nat
        ⊢ SModEq (HSMul.hSMul (HPow.hPow Top.top n) Top.top) (f n) 0
      -/
      rw [Ideal.top_pow, top_smul]
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        f : Nat → M
        x✝ : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow Top.top m) Top. …
        n : Nat
        ⊢ SModEq Top.top (f n) 0
      -/
      exact SModEq.top⟩⟩
      /-
        🎉 no goals
      -/


instance (priority := 100) of_subsingleton [Subsingleton M] : IsPrecomplete I M :=
                              /-
                                R : Type u_1
                                S : Type u_2
                                T : Type u_3
                                inst✝⁵ : CommRing R
                                I : Ideal R
                                M : Type u_4
                                inst✝⁴ : AddCommGroup M
                                inst✝³ : Module R M
                                N : Type u_5
                                inst✝² : AddCommGroup N
                                inst✝¹ : Module R N
                                inst✝ : Subsingleton M
                                f : Nat → M
                                x✝ : ∀ {m n : Nat}, LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
                                n : Nat
                                ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) 0
                              -/
  ⟨fun f _ => ⟨0, fun n => by rw [Subsingleton.elim (f n) 0]⟩⟩
                              /-
                                🎉 no goals
                              -/


/-- `AdicCompletion` is the submodule of compatible families in
`∀ n : ℕ, M ⧸ (I ^ n • ⊤)`. -/
def submodule : Submodule R (∀ n : ℕ, M ⧸ (I ^ n • ⊤ : Submodule R M)) where
  carrier := { f | ∀ {m n} (hmn : m ≤ n), AdicCompletion.transitionMap I M hmn (f n) = f m }
                      /-
                        R : Type u_1
                        S : Type u_2
                        T : Type u_3
                        inst✝⁴ : CommRing R
                        I : Ideal R
                        M : Type u_4
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        N : Type u_5
                        inst✝¹ : AddCommGroup N
                        inst✝ : Module R N
                        m✝ n✝ : Nat
                        hmn : LE.le m✝ n✝
                        ⊢ Eq ((AdicCompletion.transitionMap I M hmn) (0 n✝)) (0 m✝)
                      -/
  zero_mem' hmn := by rw [Pi.zero_apply, Pi.zero_apply, LinearMap.map_zero]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      a✝ b✝ : (n : Nat) → HasQuotient.Quotient M (HSMul.hSMul (HPow.hPow I n) Top.top)
      hf : Membership.mem (setOf fun f => ∀ {m n : Nat} (hmn : LE.le m n), Eq ((Adic …
      hg : Membership.mem (setOf fun f => ∀ {m n : Nat} (hmn : LE.le m n), Eq ((Adic …
      m n : Nat
      hmn : LE.le m n
      ⊢ Eq ((AdicCompletion.transitionMap I M hmn) (HAdd.hAdd a✝ b✝ n)) (HAdd.hAdd a …
    -/
                      /-
                        🎉 no goals
                      -/
    /-
      🎉 no goals
    -/
  add_mem' hf hg m n hmn := by
    rw [Pi.add_apply, Pi.add_apply, LinearMap.map_add, hf hmn, hg hmn]
                                 /-
                                   R : Type u_1
                                   S : Type u_2
                                   T : Type u_3
                                   inst✝⁴ : CommRing R
                                   I : Ideal R
                                   M : Type u_4
                                   inst✝³ : AddCommGroup M
                                   inst✝² : Module R M
                                   N : Type u_5
                                   inst✝¹ : AddCommGroup N
                                   inst✝ : Module R N
                                   c : R
                                   f : (n : Nat) → HasQuotient.Quotient M (HSMul.hSMul (HPow.hPow I n) Top.top)
                                   hf : Membership.mem { carrier := setOf fun f => ∀ {m n : Nat} (hmn : LE.le m n …
                                   m n : Nat
                                   hmn : LE.le m n
                                   ⊢ Eq ((AdicCompletion.transitionMap I M hmn) (HSMul.hSMul c f n)) (HSMul.hSMul …
                                 -/
  smul_mem' c f hf m n hmn := by rw [Pi.smul_apply, Pi.smul_apply, LinearMap.map_smul, hf hmn]
                                 /-
                                   🎉 no goals
                                 -/


instance : Zero (AdicCompletion I M) where
                 /-
                   R : Type u_1
                   S : Type u_2
                   T : Type u_3
                   inst✝⁴ : CommRing R
                   I : Ideal R
                   M : Type u_4
                   inst✝³ : AddCommGroup M
                   inst✝² : Module R M
                   N : Type u_5
                   inst✝¹ : AddCommGroup N
                   inst✝ : Module R N
                   ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I M hmn)  …
                 -/
  zero := ⟨0, by simp⟩
                 /-
                   🎉 no goals
                 -/


instance : Add (AdicCompletion I M) where
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  T : Type u_3
                                  inst✝⁴ : CommRing R
                                  I : Ideal R
                                  M : Type u_4
                                  inst✝³ : AddCommGroup M
                                  inst✝² : Module R M
                                  N : Type u_5
                                  inst✝¹ : AddCommGroup N
                                  inst✝ : Module R N
                                  x y : AdicCompletion I M
                                  ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I M hmn)  …
                                -/
  add x y := ⟨x.val + y.val, by simp [x.property, y.property]⟩
                                /-
                                  🎉 no goals
                                -/


instance : Neg (AdicCompletion I M) where
                        /-
                          R : Type u_1
                          S : Type u_2
                          T : Type u_3
                          inst✝⁴ : CommRing R
                          I : Ideal R
                          M : Type u_4
                          inst✝³ : AddCommGroup M
                          inst✝² : Module R M
                          N : Type u_5
                          inst✝¹ : AddCommGroup N
                          inst✝ : Module R N
                          x : AdicCompletion I M
                          ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I M hmn)  …
                        -/
  neg x := ⟨- x.val, by simp [x.property]⟩
                        /-
                          🎉 no goals
                        -/


instance : Sub (AdicCompletion I M) where
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  T : Type u_3
                                  inst✝⁴ : CommRing R
                                  I : Ideal R
                                  M : Type u_4
                                  inst✝³ : AddCommGroup M
                                  inst✝² : Module R M
                                  N : Type u_5
                                  inst✝¹ : AddCommGroup N
                                  inst✝ : Module R N
                                  x y : AdicCompletion I M
                                  ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I M hmn)  …
                                -/
  sub x y := ⟨x.val - y.val, by simp [x.property, y.property]⟩
                                /-
                                  🎉 no goals
                                -/


instance instSMul [SMul S R] [SMul S M] [IsScalarTower S R M] : SMul S (AdicCompletion I M) where
                             /-
                               R : Type u_1
                               S : Type u_2
                               T : Type u_3
                               inst✝⁷ : CommRing R
                               I : Ideal R
                               M : Type u_4
                               inst✝⁶ : AddCommGroup M
                               inst✝⁵ : Module R M
                               N : Type u_5
                               inst✝⁴ : AddCommGroup N
                               inst✝³ : Module R N
                               inst✝² : SMul S R
                               inst✝¹ : SMul S M
                               inst✝ : IsScalarTower S R M
                               r : S
                               x : AdicCompletion I M
                               ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I M hmn)  …
                             -/
  smul r x := ⟨r • x.val, by simp [x.property]⟩
                             /-
                               🎉 no goals
                             -/


@[simp, norm_cast] lemma val_zero : (0 : AdicCompletion I M).val = 0 := rfl


lemma val_zero_apply (n : ℕ) : (0 : AdicCompletion I M).val n = 0 := rfl


@[simp, norm_cast] lemma val_add (f g : AdicCompletion I M) : (f + g).val = f.val + g.val := rfl

@[simp, norm_cast] lemma val_sub (f g : AdicCompletion I M) : (f - g).val = f.val - g.val := rfl

@[simp, norm_cast] lemma val_neg (f : AdicCompletion I M) : (-f).val = -f.val := rfl


lemma val_add_apply (f g : AdicCompletion I M) (n : ℕ) : (f + g).val n = f.val n + g.val n := rfl

lemma val_sub_apply (f g : AdicCompletion I M) (n : ℕ) : (f - g).val n = f.val n - g.val n := rfl

lemma val_neg_apply (f : AdicCompletion I M) (n : ℕ) : (-f).val n = -f.val n := rfl

/- No `simp` attribute, since it causes `simp` unification timeouts when considering
the `Module (AdicCompletion I R) (AdicCompletion I M)` instance (see `AdicCompletion/Algebra`). -/

@[norm_cast]
lemma val_smul [SMul S R] [SMul S M] [IsScalarTower S R M] (s : S) (f : AdicCompletion I M) :
    (s • f).val = s • f.val := rfl


lemma val_smul_apply [SMul S R] [SMul S M] [IsScalarTower S R M] (s : S) (f : AdicCompletion I M)
    (n : ℕ) : (s • f).val n = s • f.val n := rfl


@[ext]
lemma ext {x y : AdicCompletion I M} (h : ∀ n, x.val n = y.val n) : x = y := Subtype.eq <| funext h


instance : AddCommGroup (AdicCompletion I M) :=
  let f : AdicCompletion I M → ∀ n, M ⧸ (I ^ n • ⊤ : Submodule R M) := Subtype.val
  Subtype.val_injective.addCommGroup f rfl val_add val_neg val_sub (fun _ _ ↦ val_smul ..)
    (fun _ _ ↦ val_smul ..)


instance [Semiring S] [SMul S R] [Module S M] [IsScalarTower S R M] :
    Module S (AdicCompletion I M) :=
  let f : AdicCompletion I M →+ ∀ n, M ⧸ (I ^ n • ⊤ : Submodule R M) :=
    { toFun := Subtype.val, map_zero' := rfl, map_add' := fun _ _ ↦ rfl }
  Subtype.val_injective.module S f val_smul


instance instIsScalarTower [SMul S T] [SMul S R] [SMul T R] [SMul S M] [SMul T M]
    [IsScalarTower S R M] [IsScalarTower T R M] [IsScalarTower S T M] :
    IsScalarTower S T (AdicCompletion I M) where
                         /-
                           R : Type u_1
                           S : Type u_2
                           T : Type u_3
                           inst✝¹² : CommRing R
                           I : Ideal R
                           M : Type u_4
                           inst✝¹¹ : AddCommGroup M
                           inst✝¹⁰ : Module R M
                           N : Type u_5
                           inst✝⁹ : AddCommGroup N
                           inst✝⁸ : Module R N
                           inst✝⁷ : SMul S T
                           inst✝⁶ : SMul S R
                           inst✝⁵ : SMul T R
                           inst✝⁴ : SMul S M
                           inst✝³ : SMul T M
                           inst✝² : IsScalarTower S R M
                           inst✝¹ : IsScalarTower T R M
                           inst✝ : IsScalarTower S T M
                           s : S
                           t : T
                           f : AdicCompletion I M
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul s t) f) (HSMul.hSMul s (HSMul.hSMul t f))
                         -/
  smul_assoc s t f := by ext; simp [val_smul]
                              /-
                                🎉 no goals
                              -/


instance instSMulCommClass [SMul S R] [SMul T R] [SMul S M] [SMul T M]
    [IsScalarTower S R M] [IsScalarTower T R M] [SMulCommClass S T M] :
    SMulCommClass S T (AdicCompletion I M) where
                        /-
                          R : Type u_1
                          S : Type u_2
                          T : Type u_3
                          inst✝¹¹ : CommRing R
                          I : Ideal R
                          M : Type u_4
                          inst✝¹⁰ : AddCommGroup M
                          inst✝⁹ : Module R M
                          N : Type u_5
                          inst✝⁸ : AddCommGroup N
                          inst✝⁷ : Module R N
                          inst✝⁶ : SMul S R
                          inst✝⁵ : SMul T R
                          inst✝⁴ : SMul S M
                          inst✝³ : SMul T M
                          inst✝² : IsScalarTower S R M
                          inst✝¹ : IsScalarTower T R M
                          inst✝ : SMulCommClass S T M
                          s : S
                          t : T
                          f : AdicCompletion I M
                          ⊢ Eq (HSMul.hSMul s (HSMul.hSMul t f)) (HSMul.hSMul t (HSMul.hSMul s f))
                        -/
  smul_comm s t f := by ext; simp [val_smul, smul_comm]
                             /-
                               🎉 no goals
                             -/


instance instIsCentralScalar [SMul S R] [SMul Sᵐᵒᵖ R] [SMul S M] [SMul Sᵐᵒᵖ M]
    [IsScalarTower S R M] [IsScalarTower Sᵐᵒᵖ R M] [IsCentralScalar S M] :
    IsCentralScalar S (AdicCompletion I M) where
                            /-
                              R : Type u_1
                              S : Type u_2
                              T : Type u_3
                              inst✝¹¹ : CommRing R
                              I : Ideal R
                              M : Type u_4
                              inst✝¹⁰ : AddCommGroup M
                              inst✝⁹ : Module R M
                              N : Type u_5
                              inst✝⁸ : AddCommGroup N
                              inst✝⁷ : Module R N
                              inst✝⁶ : SMul S R
                              inst✝⁵ : SMul (MulOpposite S) R
                              inst✝⁴ : SMul S M
                              inst✝³ : SMul (MulOpposite S) M
                              inst✝² : IsScalarTower S R M
                              inst✝¹ : IsScalarTower (MulOpposite S) R M
                              inst✝ : IsCentralScalar S M
                              s : S
                              f : AdicCompletion I M
                              ⊢ Eq (HSMul.hSMul (MulOpposite.op s) f) (HSMul.hSMul s f)
                            -/
  op_smul_eq_smul s f := by ext; simp [val_smul, op_smul_eq_smul]
                                 /-
                                   🎉 no goals
                                 -/


/-- The canonical inclusion from the completion to the product. -/
@[simps]
def incl : AdicCompletion I M →ₗ[R] (∀ n, M ⧸ (I ^ n • ⊤ : Submodule R M)) where
  toFun x := x.val
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp, norm_cast]
lemma val_sum {ι : Type*} (s : Finset ι) (f : ι → AdicCompletion I M) :
    (∑ i ∈ s, f i).val = ∑ i ∈ s, (f i).val := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_6
    s : Finset ι
    f : ι → AdicCompletion I M
    ⊢ Eq (↑(s.sum fun i => f i)) (s.sum fun i => ↑(f i))
  -/
  simp_rw [← funext (incl_apply _ _ _), map_sum]
  /-
    🎉 no goals
  -/


lemma val_sum_apply {ι : Type*} (s : Finset ι) (f : ι → AdicCompletion I M) (n : ℕ) :
                                                      /-
                                                        R : Type u_1
                                                        inst✝² : CommRing R
                                                        I : Ideal R
                                                        M : Type u_4
                                                        inst✝¹ : AddCommGroup M
                                                        inst✝ : Module R M
                                                        ι : Type u_6
                                                        s : Finset ι
                                                        f : ι → AdicCompletion I M
                                                        n : Nat
                                                        ⊢ Eq (↑(s.sum fun i => f i) n) (s.sum fun i => ↑(f i) n)
                                                      -/
    (∑ i ∈ s, f i).val n = ∑ i ∈ s, (f i).val n := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The canonical linear map to the completion. -/
def of : M →ₗ[R] AdicCompletion I M where
  toFun x := ⟨fun n => mkQ (I ^ n • ⊤ : Submodule R M) x, fun _ => rfl⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem of_apply (x : M) (n : ℕ) : (of I M x).1 n = mkQ (I ^ n • ⊤ : Submodule R M) x :=
  rfl


/-- Linearly evaluating a sequence in the completion at a given input. -/
def eval (n : ℕ) : AdicCompletion I M →ₗ[R] M ⧸ (I ^ n • ⊤ : Submodule R M) where
  toFun f := f.1 n
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem coe_eval (n : ℕ) :
    (eval I M n : AdicCompletion I M → M ⧸ (I ^ n • ⊤ : Submodule R M)) = fun f => f.1 n :=
  rfl


theorem eval_apply (n : ℕ) (f : AdicCompletion I M) : eval I M n f = f.1 n :=
  rfl


theorem eval_of (n : ℕ) (x : M) : eval I M n (of I M x) = mkQ (I ^ n • ⊤ : Submodule R M) x :=
  rfl


@[simp]
theorem eval_comp_of (n : ℕ) : (eval I M n).comp (of I M) = mkQ _ :=
  rfl


theorem eval_surjective (n : ℕ) : Function.Surjective (eval I M n) := fun x ↦
  Quotient.inductionOn' x fun x ↦ ⟨of I M x, rfl⟩


@[simp]
theorem range_eval (n : ℕ) : LinearMap.range (eval I M n) = ⊤ :=
  LinearMap.range_eq_top.2 (eval_surjective I M n)


instance : IsHausdorff I (AdicCompletion I M) where
  haus' x h := ext fun n ↦ by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      x : AdicCompletion I M
      h : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) x 0
      n : Nat
      ⊢ Eq (↑x n) (↑0 n)
    -/
    refine smul_induction_on (SModEq.zero.1 <| h n) (fun r hr x _ ↦ ?_) (fun x y hx hy ↦ ?_)
      /-
        case refine_1
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        x✝¹ : AdicCompletion I M
        h : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) x✝¹ 0
        n : Nat
        r : R
        hr : Membership.mem (HPow.hPow I n) r
        x : AdicCompletion I M
        x✝ : Membership.mem Top.top x
        ⊢ Eq (↑(HSMul.hSMul r x) n) (↑0 n)
      -/
    · simp only [val_smul_apply, val_zero]
      exact Quotient.inductionOn' (x.val n)
        (fun a ↦ SModEq.zero.2 <| smul_mem_smul hr mem_top)
      /-
        case refine_2
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommRing R
        I : Ideal R
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        N : Type u_5
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        x✝ : AdicCompletion I M
        h : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) x✝ 0
        n : Nat
        x y : AdicCompletion I M
        hx : Eq (↑x n) (↑0 n)
        hy : Eq (↑y n) (↑0 n)
        ⊢ Eq (↑(HAdd.hAdd x y) n) (↑0 n)
      -/
    · simp only [val_add_apply, hx, val_zero_apply, hy, add_zero]
      /-
        🎉 no goals
      -/


@[simp]
theorem transitionMap_mk {m n : ℕ} (hmn : m ≤ n) (x : M) :
    transitionMap I M hmn
      (Submodule.Quotient.mk (p := (I ^ n • ⊤ : Submodule R M)) x) =
      Submodule.Quotient.mk (p := (I ^ m • ⊤ : Submodule R M)) x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n : Nat
    hmn : LE.le m n
    x : M
    ⊢ Eq ((AdicCompletion.transitionMap I M hmn) (Submodule.Quotient.mk x)) (Submo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem transitionMap_eq (n : ℕ) : transitionMap I M (Nat.le_refl n) = LinearMap.id := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ Eq (AdicCompletion.transitionMap I M ⋯) LinearMap.id
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    x✝ : M
    ⊢ Eq (((AdicCompletion.transitionMap I M ⋯).comp (HSMul.hSMul (HPow.hPow I n)  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem transitionMap_comp {m n k : ℕ} (hmn : m ≤ n) (hnk : n ≤ k) :
    transitionMap I M hmn ∘ₗ transitionMap I M hnk = transitionMap I M (hmn.trans hnk) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    ⊢ Eq ((AdicCompletion.transitionMap I M hmn).comp (AdicCompletion.transitionMa …
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    x✝ : M
    ⊢ Eq ((((AdicCompletion.transitionMap I M hmn).comp (AdicCompletion.transition …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem transitionMap_comp_apply {m n k : ℕ} (hmn : m ≤ n) (hnk : n ≤ k)
    (x : M ⧸ (I ^ k • ⊤ : Submodule R M)) :
    transitionMap I M hmn (transitionMap I M hnk x) = transitionMap I M (hmn.trans hnk) x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    x : HasQuotient.Quotient M (HSMul.hSMul (HPow.hPow I k) Top.top)
    ⊢ Eq ((AdicCompletion.transitionMap I M hmn) ((AdicCompletion.transitionMap I  …
  -/
  change (transitionMap I M hmn ∘ₗ transitionMap I M hnk) x = transitionMap I M (hmn.trans hnk) x
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    x : HasQuotient.Quotient M (HSMul.hSMul (HPow.hPow I k) Top.top)
    ⊢ Eq (((AdicCompletion.transitionMap I M hmn).comp (AdicCompletion.transitionM …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem transitionMap_comp_eval_apply {m n : ℕ} (hmn : m ≤ n) (x : AdicCompletion I M) :
    transitionMap I M hmn (x.val n) = x.val m :=
  x.property hmn


@[simp]
theorem transitionMap_comp_eval {m n : ℕ} (hmn : m ≤ n) :
    transitionMap I M hmn ∘ₗ eval I M n = eval I M m := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq ((AdicCompletion.transitionMap I M hmn).comp (AdicCompletion.eval I M n)) …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m n : Nat
    hmn : LE.le m n
    x : AdicCompletion I M
    ⊢ Eq (((AdicCompletion.transitionMap I M hmn).comp (AdicCompletion.eval I M n) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A sequence `ℕ → M` is an `I`-adic Cauchy sequence if for every `m ≤ n`,
`f m ≡ f n` modulo `I ^ m • ⊤`. -/
def IsAdicCauchy (f : ℕ → M) : Prop :=
  ∀ {m n}, m ≤ n → f m ≡ f n [SMOD (I ^ m • ⊤ : Submodule R M)]


/-- The type of `I`-adic Cauchy sequences. -/
def AdicCauchySequence : Type _ := { f : ℕ → M // IsAdicCauchy I M f }


/-- The type of `I`-adic cauchy sequences is a submodule of the product `ℕ → M`. -/
def submodule : Submodule R (ℕ → M) where
  carrier := { f | IsAdicCauchy I M f }
  add_mem' := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ ∀ {a b : Nat → M}, Membership.mem (setOf fun f => AdicCompletion.IsAdicCauch …
    -/
    intro f g hf hg m n hmn
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f g : Nat → M
      hf : Membership.mem (setOf fun f => AdicCompletion.IsAdicCauchy I M f) f
      hg : Membership.mem (setOf fun f => AdicCompletion.IsAdicCauchy I M f) g
      m n : Nat
      hmn : LE.le m n
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) (HAdd.hAdd f g m) (HAdd.hAdd f  …
    -/
    exact SModEq.add (hf hmn) (hg hmn)
    /-
      🎉 no goals
    -/
  zero_mem' := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ Membership.mem { carrier := setOf fun f => AdicCompletion.IsAdicCauchy I M f …
    -/
    intro _ _ _
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      m✝ n✝ : Nat
      a✝ : LE.le m✝ n✝
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I m✝) Top.top) (0 m✝) (0 n✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  smul_mem' := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ ∀ (c : R) {x : Nat → M}, Membership.mem { carrier := setOf fun f => AdicComp …
    -/
    intro r f hf m n hmn
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      r : R
      f : Nat → M
      hf : Membership.mem { carrier := setOf fun f => AdicCompletion.IsAdicCauchy I  …
      m n : Nat
      hmn : LE.le m n
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) (HSMul.hSMul r f m) (HSMul.hSMu …
    -/
    exact SModEq.smul (hf hmn) r
    /-
      🎉 no goals
    -/


instance : Zero (AdicCauchySequence I M) where
  zero := ⟨0, fun _ ↦ rfl⟩


instance : Add (AdicCauchySequence I M) where
  add x y := ⟨x.val + y.val, fun hmn ↦ SModEq.add (x.property hmn) (y.property hmn)⟩


instance : Neg (AdicCauchySequence I M) where
  neg x := ⟨- x.val, fun hmn ↦ SModEq.neg (x.property hmn)⟩


instance : Sub (AdicCauchySequence I M) where
  sub x y := ⟨x.val - y.val, fun hmn ↦ SModEq.sub (x.property hmn) (y.property hmn)⟩


instance : SMul ℕ (AdicCauchySequence I M) where
  smul n x := ⟨n • x.val, fun hmn ↦ SModEq.nsmul (x.property hmn) n⟩


instance : SMul ℤ (AdicCauchySequence I M) where
  smul n x := ⟨n • x.val, fun hmn ↦ SModEq.zsmul (x.property hmn) n⟩


instance : AddCommGroup (AdicCauchySequence I M) := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ AddCommGroup (AdicCompletion.AdicCauchySequence I M)
  -/
  let f : AdicCauchySequence I M → (ℕ → M) := Subtype.val
  apply Subtype.val_injective.addCommGroup f rfl (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)


instance : SMul R (AdicCauchySequence I M) where
  smul r x := ⟨r • x.val, fun hmn ↦ SModEq.smul (x.property hmn) r⟩


instance : Module R (AdicCauchySequence I M) :=
  let f : AdicCauchySequence I M →+ (ℕ → M) :=
    { toFun := Subtype.val, map_zero' := rfl, map_add' := fun _ _ ↦ rfl }
  Subtype.val_injective.module R f (fun _ _ ↦ rfl)


instance : CoeFun (AdicCauchySequence I M) (fun _ ↦ ℕ → M) where
  coe f := f.val


@[simp]
theorem zero_apply (n : ℕ) : (0 : AdicCauchySequence I M) n = 0 :=
  rfl


@[simp]
theorem add_apply (n : ℕ) (f g : AdicCauchySequence I M) : (f + g) n = f n + g n :=
  rfl


@[simp]
theorem sub_apply (n : ℕ) (f g : AdicCauchySequence I M) : (f - g) n = f n - g n :=
  rfl


@[simp]
theorem smul_apply (n : ℕ) (r : R) (f : AdicCauchySequence I M) : (r • f) n = r • f n :=
  rfl


@[ext]
theorem ext {x y : AdicCauchySequence I M} (h : ∀ n, x n = y n) : x = y :=
  Subtype.eq <| funext h


/-- The defining property of an adic cauchy sequence unwrapped. -/
theorem mk_eq_mk {m n : ℕ} (hmn : m ≤ n) (f : AdicCauchySequence I M) :
    Submodule.Quotient.mk (p := (I ^ m • ⊤ : Submodule R M)) (f n) =
      Submodule.Quotient.mk (p := (I ^ m • ⊤ : Submodule R M)) (f m) :=
  (f.property hmn).symm


/-- The `I`-adic cauchy condition can be checked on successive `n`.-/
theorem isAdicCauchy_iff (f : ℕ → M) :
    IsAdicCauchy I M f ↔ ∀ n, f n ≡ f (n + 1) [SMOD (I ^ n • ⊤ : Submodule R M)] := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Nat → M
    ⊢ Iff (AdicCompletion.IsAdicCauchy I M f) (∀ (n : Nat), SModEq (HSMul.hSMul (H …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Nat → M
      ⊢ AdicCompletion.IsAdicCauchy I M f → ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.h …
    -/
  · intro h n
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Nat → M
      h : AdicCompletion.IsAdicCauchy I M f
      n : Nat
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) (f (HAdd.hAdd n 1))
    -/
    exact h (Nat.le_succ n)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Nat → M
      ⊢ (∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) (f (HAdd.hA …
    -/
  · intro h m n hmn
    induction n, hmn using Nat.le_induction with
    | base => rfl
    | succ n hmn ih =>
        trans
        · exact ih
        · refine SModEq.mono (smul_mono (Ideal.pow_le_pow_right hmn) (by rfl)) (h n)


/-- Construct `I`-adic cauchy sequence from sequence satisfying the successive cauchy condition. -/
@[simps]
def AdicCauchySequence.mk (f : ℕ → M)
    (h : ∀ n, f n ≡ f (n + 1) [SMOD (I ^ n • ⊤ : Submodule R M)]) : AdicCauchySequence I M where
  val := f
                 /-
                   R : Type u_1
                   S : Type u_2
                   T : Type u_3
                   inst✝⁴ : CommRing R
                   I : Ideal R
                   M : Type u_4
                   inst✝³ : AddCommGroup M
                   inst✝² : Module R M
                   N : Type u_5
                   inst✝¹ : AddCommGroup N
                   inst✝ : Module R N
                   f : Nat → M
                   h : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) (f (HAdd.h …
                   ⊢ AdicCompletion.IsAdicCauchy I M f
                 -/
  property := by rwa [isAdicCauchy_iff]
                 /-
                   🎉 no goals
                 -/


/-- The canonical linear map from cauchy sequences to the completion. -/
@[simps]
def mk : AdicCauchySequence I M →ₗ[R] AdicCompletion I M where
  toFun f := ⟨fun n ↦ Submodule.mkQ (I ^ n • ⊤ : Submodule R M) (f n), by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : AdicCompletion.AdicCauchySequence I M
      ⊢ ∀ {m n : Nat} (hmn : LE.le m n), Eq ((AdicCompletion.transitionMap I M hmn)  …
    -/
    intro m n hmn
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : AdicCompletion.AdicCauchySequence I M
      m n : Nat
      hmn : LE.le m n
      ⊢ Eq ((AdicCompletion.transitionMap I M hmn) ((fun n => (HSMul.hSMul (HPow.hPo …
    -/
    simp only [mkQ_apply, transitionMap_mk]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : AdicCompletion.AdicCauchySequence I M
      m n : Nat
      hmn : LE.le m n
      ⊢ Eq (Submodule.Quotient.mk (↑f n)) (Submodule.Quotient.mk (↑f m))
    -/
    exact (f.property hmn).symm⟩
    /-
      🎉 no goals
    -/
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- Criterion for checking that an adic cauchy sequence is mapped to zero in the adic completion. -/
theorem mk_zero_of (f : AdicCauchySequence I M)
    (h : ∃ k : ℕ, ∀ n ≥ k, ∃ m ≥ n, ∃ l ≥ n, f m ∈ (I ^ l • ⊤ : Submodule R M)) :
    AdicCompletion.mk I M f = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : AdicCompletion.AdicCauchySequence I M
    h : Exists fun k => ∀ (n : Nat), GE.ge n k → Exists fun m => And (GE.ge m n) ( …
    ⊢ Eq ((AdicCompletion.mk I M) f) 0
  -/
  obtain ⟨k, h⟩ := h
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : AdicCompletion.AdicCauchySequence I M
    k : Nat
    h : ∀ (n : Nat), GE.ge n k → Exists fun m => And (GE.ge m n) (Exists fun l =>  …
    ⊢ Eq ((AdicCompletion.mk I M) f) 0
  -/
  ext n
  /-
    case intro.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : AdicCompletion.AdicCauchySequence I M
    k : Nat
    h : ∀ (n : Nat), GE.ge n k → Exists fun m => And (GE.ge m n) (Exists fun l =>  …
    n : Nat
    ⊢ Eq (↑((AdicCompletion.mk I M) f) n) (↑0 n)
  -/
  obtain ⟨m, hnm, l, hnl, hl⟩ := h (n + k) (by omega)
  rw [mk_apply_coe, Submodule.mkQ_apply, val_zero,
    ← AdicCauchySequence.mk_eq_mk (show n ≤ m by omega)]
  /-
    case intro.h.intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : AdicCompletion.AdicCauchySequence I M
    k : Nat
    h : ∀ (n : Nat), GE.ge n k → Exists fun m => And (GE.ge m n) (Exists fun l =>  …
    n m : Nat
    hnm : GE.ge m (HAdd.hAdd n k)
    l : Nat
    hnl : GE.ge l (HAdd.hAdd n k)
    hl : Membership.mem (HSMul.hSMul (HPow.hPow I l) Top.top) (↑f m)
    ⊢ Eq (Submodule.Quotient.mk (↑f m)) (0 n)
  -/
  simpa using (Submodule.smul_mono_left (Ideal.pow_le_pow_right (by omega))) hl
  /-
    🎉 no goals
  -/


/-- Every element in the adic completion is represented by a Cauchy sequence. -/
theorem mk_surjective : Function.Surjective (mk I M) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Function.Surjective ⇑(AdicCompletion.mk I M)
  -/
  intro x
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : AdicCompletion I M
    ⊢ Exists fun a => Eq ((AdicCompletion.mk I M) a) x
  -/
  choose a ha using fun n ↦ Submodule.Quotient.mk_surjective _ (x.val n)
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : AdicCompletion I M
    a : Nat → M
    ha : ∀ (n : Nat), Eq (Submodule.Quotient.mk (a n)) (↑x n)
    ⊢ Exists fun a => Eq ((AdicCompletion.mk I M) a) x
  -/
  refine ⟨⟨a, ?_⟩, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : AdicCompletion I M
      a : Nat → M
      ha : ∀ (n : Nat), Eq (Submodule.Quotient.mk (a n)) (↑x n)
      ⊢ AdicCompletion.IsAdicCauchy I M a
    -/
  · intro m n hmn
    /-
      case refine_1
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : AdicCompletion I M
      a : Nat → M
      ha : ∀ (n : Nat), Eq (Submodule.Quotient.mk (a n)) (↑x n)
      m n : Nat
      hmn : LE.le m n
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) (a m) (a n)
    -/
    rw [SModEq.def, ha m, ← transitionMap_mk I M hmn, ha n, x.property hmn]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : AdicCompletion I M
      a : Nat → M
      ha : ∀ (n : Nat), Eq (Submodule.Quotient.mk (a n)) (↑x n)
      ⊢ Eq ((AdicCompletion.mk I M) ⟨a, ⋯⟩) x
    -/
  · ext n
    /-
      case refine_2.h
      R : Type u_1
      inst✝² : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : AdicCompletion I M
      a : Nat → M
      ha : ∀ (n : Nat), Eq (Submodule.Quotient.mk (a n)) (↑x n)
      n : Nat
      ⊢ Eq (↑((AdicCompletion.mk I M) ⟨a, ⋯⟩) n) (↑x n)
    -/
    simp [ha n]
    /-
      🎉 no goals
    -/


/-- To show a statement about an element of `adicCompletion I M`, it suffices to check it
on Cauchy sequences. -/
theorem induction_on {p : AdicCompletion I M → Prop} (x : AdicCompletion I M)
    (h : ∀ (f : AdicCauchySequence I M), p (mk I M f)) : p x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : AdicCompletion I M → Prop
    x : AdicCompletion I M
    h : ∀ (f : AdicCompletion.AdicCauchySequence I M), p ((AdicCompletion.mk I M) f)
    ⊢ p x
  -/
  obtain ⟨f, rfl⟩ := mk_surjective I M x
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : AdicCompletion I M → Prop
    h : ∀ (f : AdicCompletion.AdicCauchySequence I M), p ((AdicCompletion.mk I M) f)
    f : AdicCompletion.AdicCauchySequence I M
    ⊢ p ((AdicCompletion.mk I M) f)
  -/
  exact h f
  /-
    🎉 no goals
  -/


/-- Lift a compatible family of linear maps `M →ₗ[R] N ⧸ (I ^ n • ⊤ : Submodule R N)` to
the `I`-adic completion of `M`. -/
def lift (f : ∀ (n : ℕ), M →ₗ[R] N ⧸ (I ^ n • ⊤ : Submodule R N))
    (h : ∀ {m n : ℕ} (hle : m ≤ n), transitionMap I N hle ∘ₗ f n = f m) :
    M →ₗ[R] AdicCompletion I N where
  toFun := fun x ↦ ⟨fun n ↦ f n x, fun hkl ↦ LinearMap.congr_fun (h hkl) x⟩
  map_add' x y := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : (n : Nat) → LinearMap (RingHom.id R) M (HasQuotient.Quotient N (HSMul.hSMu …
      h : ∀ {m n : Nat} (hle : LE.le m n), Eq ((AdicCompletion.transitionMap I N hle …
      x y : M
      ⊢ Eq ((fun x => ⟨fun n => (f n) x, ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x =>  …
    -/
    simp only [map_add]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : (n : Nat) → LinearMap (RingHom.id R) M (HasQuotient.Quotient N (HSMul.hSMu …
      h : ∀ {m n : Nat} (hle : LE.le m n), Eq ((AdicCompletion.transitionMap I N hle …
      x y : M
      ⊢ Eq ⟨fun n => HAdd.hAdd ((f n) x) ((f n) y), ⋯⟩ (HAdd.hAdd ⟨fun n => (f n) x, …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_smul' r x := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : (n : Nat) → LinearMap (RingHom.id R) M (HasQuotient.Quotient N (HSMul.hSMu …
      h : ∀ {m n : Nat} (hle : LE.le m n), Eq ((AdicCompletion.transitionMap I N hle …
      r : R
      x : M
      ⊢ Eq ({ toFun := fun x => ⟨fun n => (f n) x, ⋯⟩, map_add' := ⋯ }.toFun (HSMul. …
    -/
    simp only [LinearMapClass.map_smul, RingHom.id_apply]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u_5
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : (n : Nat) → LinearMap (RingHom.id R) M (HasQuotient.Quotient N (HSMul.hSMu …
      h : ∀ {m n : Nat} (hle : LE.le m n), Eq ((AdicCompletion.transitionMap I N hle …
      r : R
      x : M
      ⊢ Eq ⟨fun n => HSMul.hSMul r ((f n) x), ⋯⟩ (HSMul.hSMul r ⟨fun n => (f n) x, ⋯⟩)
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
lemma eval_lift (f : ∀ (n : ℕ), M →ₗ[R] N ⧸ (I ^ n • ⊤ : Submodule R N))
    (h : ∀ {m n : ℕ} (hle : m ≤ n), transitionMap I N hle ∘ₗ f n = f m)
    (n : ℕ) : eval I N n ∘ₗ lift I f h = f n :=
  rfl


@[simp]
lemma eval_lift_apply (f : ∀ (n : ℕ), M →ₗ[R] N ⧸ (I ^ n • ⊤ : Submodule R N))
    (h : ∀ {m n : ℕ} (hle : m ≤ n), transitionMap I N hle ∘ₗ f n = f m)
    (n : ℕ) (x : M) : (lift I f h x).val n = f n x :=
  rfl


instance bot : IsAdicComplete (⊥ : Ideal R) M where


protected theorem subsingleton (h : IsAdicComplete (⊤ : Ideal R) M) : Subsingleton M :=
  h.1.subsingleton


instance (priority := 100) of_subsingleton [Subsingleton M] : IsAdicComplete I M where


theorem le_jacobson_bot [IsAdicComplete I R] : I ≤ (⊥ : Ideal R).jacobson := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    ⊢ LE.le I Bot.bot.jacobson
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    ⊢ Membership.mem Bot.bot.jacobson x
  -/
  rw [← Ideal.neg_mem_iff, Ideal.mem_jacobson_bot]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    ⊢ ∀ (y : R), IsUnit (HAdd.hAdd (HMul.hMul (Neg.neg x) y) 1)
  -/
  intro y
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    ⊢ IsUnit (HAdd.hAdd (HMul.hMul (Neg.neg x) y) 1)
  -/
  rw [add_comm]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    ⊢ IsUnit (HAdd.hAdd 1 (HMul.hMul (Neg.neg x) y))
  -/
  let f : ℕ → R := fun n => ∑ i ∈ range n, (x * y) ^ i
  have hf : ∀ m n, m ≤ n → f m ≡ f n [SMOD I ^ m • (⊤ : Submodule R R)] := by
    intro m n h
    simp only [f, Algebra.id.smul_eq_mul, Ideal.mul_top, SModEq.sub_mem]
    rw [← add_tsub_cancel_of_le h, Finset.sum_range_add, ← sub_sub, sub_self, zero_sub,
      @neg_mem_iff]
    apply Submodule.sum_mem
    intro n _
    rw [mul_pow, pow_add, mul_assoc]
    exact Ideal.mul_mem_right _ (I ^ m) (Ideal.pow_mem_pow hx m)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    ⊢ IsUnit (HAdd.hAdd 1 (HMul.hMul (Neg.neg x) y))
  -/
  obtain ⟨L, hL⟩ := IsPrecomplete.prec toIsPrecomplete @hf
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    hL : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    ⊢ IsUnit (HAdd.hAdd 1 (HMul.hMul (Neg.neg x) y))
  -/
  rw [isUnit_iff_exists_inv]
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    hL : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    ⊢ Exists fun b => Eq (HMul.hMul (HAdd.hAdd 1 (HMul.hMul (Neg.neg x) y)) b) 1
  -/
  use L
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    hL : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    ⊢ Eq (HMul.hMul (HAdd.hAdd 1 (HMul.hMul (Neg.neg x) y)) L) 1
  -/
  rw [← sub_eq_zero, neg_mul]
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    hL : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd 1 (Neg.neg (HMul.hMul x y))) L) 1) 0
  -/
  apply IsHausdorff.haus (toIsHausdorff : IsHausdorff I R)
  /-
    case h.a
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    hL : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    ⊢ ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (HSub.hSub (HMul.h …
  -/
  intro n
  /-
    case h.a
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    hL : ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    n : Nat
    ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (HSub.hSub (HMul.hMul (HAdd.hAd …
  -/
  specialize hL n
  /-
    case h.a
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    n : Nat
    hL : SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (f n) L
    ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (HSub.hSub (HMul.hMul (HAdd.hAd …
  -/
  rw [SModEq.sub_mem, Algebra.id.smul_eq_mul, Ideal.mul_top] at hL ⊢
  /-
    case h.a
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    n : Nat
    hL : Membership.mem (HPow.hPow I n) (HSub.hSub (f n) L)
    ⊢ Membership.mem (HPow.hPow I n) (HSub.hSub (HSub.hSub (HMul.hMul (HAdd.hAdd 1 …
  -/
  rw [sub_zero]
  suffices (1 - x * y) * f n - 1 ∈ I ^ n by
    convert Ideal.sub_mem _ this (Ideal.mul_mem_left _ (1 + -(x * y)) hL) using 1
    ring
  /-
    case h.a
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsAdicComplete I R
    x : R
    hx : Membership.mem I x
    y : R
    f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
    hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
    L : R
    n : Nat
    hL : Membership.mem (HPow.hPow I n) (HSub.hSub (f n) L)
    ⊢ Membership.mem (HPow.hPow I n) (HSub.hSub (HMul.hMul (HSub.hSub 1 (HMul.hMul …
  -/
  cases n
    /-
      case h.a.zero
      R : Type u_1
      inst✝¹ : CommRing R
      I : Ideal R
      inst✝ : IsAdicComplete I R
      x : R
      hx : Membership.mem I x
      y : R
      f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
      hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
      L : R
      hL : Membership.mem (HPow.hPow I 0) (HSub.hSub (f 0) L)
      ⊢ Membership.mem (HPow.hPow I 0) (HSub.hSub (HMul.hMul (HSub.hSub 1 (HMul.hMul …
    -/
  · simp only [Ideal.one_eq_top, pow_zero, mem_top]
    /-
      🎉 no goals
    -/
  · rw [← neg_sub _ (1 : R), neg_mul, mul_geom_sum, neg_sub, sub_sub, add_comm (_ ^ _), ← sub_sub,
      sub_self, zero_sub, @neg_mem_iff, mul_pow]
    /-
      case h.a.succ
      R : Type u_1
      inst✝¹ : CommRing R
      I : Ideal R
      inst✝ : IsAdicComplete I R
      x : R
      hx : Membership.mem I x
      y : R
      f : Nat → R := fun n => (Finset.range n).sum fun i => HPow.hPow (HMul.hMul x y …
      hf : ∀ (m n : Nat), LE.le m n → SModEq (HSMul.hSMul (HPow.hPow I m) Top.top) ( …
      L : R
      n✝ : Nat
      hL : Membership.mem (HPow.hPow I (HAdd.hAdd n✝ 1)) (HSub.hSub (f (HAdd.hAdd n✝ …
      ⊢ Membership.mem (HPow.hPow I (HAdd.hAdd n✝ 1)) (HMul.hMul (HPow.hPow x (HAdd. …
    -/
    exact Ideal.mul_mem_right _ (I ^ _) (Ideal.pow_mem_pow hx _)
    /-
      🎉 no goals
    -/


