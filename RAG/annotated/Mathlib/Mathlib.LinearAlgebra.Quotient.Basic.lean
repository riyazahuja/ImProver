/-- The quotient of `P` as an `S`-submodule is the same as the quotient of `P` as an `R`-submodule,
where `P : Submodule R M`.
-/
def restrictScalarsEquiv [Ring S] [SMul S R] [Module S M] [IsScalarTower S R M]
    (P : Submodule R M) : (M ⧸ P.restrictScalars S) ≃ₗ[S] M ⧸ P :=
  { Quotient.congrRight fun _ _ => Iff.rfl with
    map_add' := fun x y => Quotient.inductionOn₂' x y fun _x' _y' => rfl
    map_smul' := fun _c x => Submodule.Quotient.induction_on _ x fun _x' => rfl }


@[simp]
theorem restrictScalarsEquiv_mk [Ring S] [SMul S R] [Module S M] [IsScalarTower S R M]
    (P : Submodule R M) (x : M) :
    restrictScalarsEquiv S P (mk x) = mk x :=
  rfl


@[simp]
theorem restrictScalarsEquiv_symm_mk [Ring S] [SMul S R] [Module S M] [IsScalarTower S R M]
    (P : Submodule R M) (x : M) :
    (restrictScalarsEquiv S P).symm (mk x) = mk x :=
  rfl


theorem nontrivial_of_lt_top (h : p < ⊤) : Nontrivial (M ⧸ p) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    h : LT.lt p Top.top
    ⊢ Nontrivial (HasQuotient.Quotient M p)
  -/
  obtain ⟨x, _, not_mem_s⟩ := SetLike.exists_of_lt h
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    h : LT.lt p Top.top
    x : M
    left✝ : Membership.mem Top.top x
    not_mem_s : Not (Membership.mem p x)
    ⊢ Nontrivial (HasQuotient.Quotient M p)
  -/
  refine ⟨⟨mk x, 0, ?_⟩⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    h : LT.lt p Top.top
    x : M
    left✝ : Membership.mem Top.top x
    not_mem_s : Not (Membership.mem p x)
    ⊢ Ne (Submodule.Quotient.mk x) 0
  -/
  simpa using not_mem_s
  /-
    🎉 no goals
  -/


instance QuotientBot.infinite [Infinite M] : Infinite (M ⧸ (⊥ : Submodule R M)) :=
  Infinite.of_injective Submodule.Quotient.mk fun _x _y h =>
    sub_eq_zero.mp <| (Submodule.Quotient.eq ⊥).mp h


instance QuotientTop.unique : Unique (M ⧸ (⊤ : Submodule R M)) where
  default := 0
  uniq x := Submodule.Quotient.induction_on _ x fun _x =>
    (Submodule.Quotient.eq ⊤).mpr Submodule.mem_top


instance QuotientTop.fintype : Fintype (M ⧸ (⊤ : Submodule R M)) :=
  Fintype.ofSubsingleton 0


theorem subsingleton_quotient_iff_eq_top : Subsingleton (M ⧸ p) ↔ p = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    ⊢ Iff (Subsingleton (HasQuotient.Quotient M p)) (Eq p Top.top)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      ⊢ Subsingleton (HasQuotient.Quotient M p) → Eq p Top.top
    -/
  · rintro h
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      h : Subsingleton (HasQuotient.Quotient M p)
      ⊢ Eq p Top.top
    -/
    refine eq_top_iff.mpr fun x _ => ?_
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      h : Subsingleton (HasQuotient.Quotient M p)
      x : M
      x✝ : Membership.mem Top.top x
      ⊢ Membership.mem p x
    -/
    have : x - 0 ∈ p := (Submodule.Quotient.eq p).mp (Subsingleton.elim _ _)
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      h : Subsingleton (HasQuotient.Quotient M p)
      x : M
      x✝ : Membership.mem Top.top x
      this : Membership.mem p (HSub.hSub x 0)
      ⊢ Membership.mem p x
    -/
    rwa [sub_zero] at this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      ⊢ Eq p Top.top → Subsingleton (HasQuotient.Quotient M p)
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      ⊢ Subsingleton (HasQuotient.Quotient M Top.top)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem unique_quotient_iff_eq_top : Nonempty (Unique (M ⧸ p)) ↔ p = ⊤ :=
  ⟨fun ⟨h⟩ => subsingleton_quotient_iff_eq_top.mp (@Unique.instSubsingleton _ h),
       /-
         R : Type u_1
         M : Type u_2
         inst✝² : Ring R
         inst✝¹ : AddCommGroup M
         inst✝ : Module R M
         p : Submodule R M
         ⊢ Eq p Top.top → Nonempty (Unique (HasQuotient.Quotient M p))
       -/
    by rintro rfl; exact ⟨QuotientTop.unique⟩⟩
                   /-
                     🎉 no goals
                   -/


noncomputable instance Quotient.fintype [Fintype M] (S : Submodule R M) : Fintype (M ⧸ S) :=
  @_root_.Quotient.fintype _ _ _ fun _ _ => Classical.dec _


theorem card_eq_card_quotient_mul_card (S : Submodule R M) :
    Nat.card M = Nat.card S * Nat.card (M ⧸ S) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    ⊢ Eq (Nat.card M) (HMul.hMul (Nat.card (Subtype fun x => Membership.mem S x))  …
  -/
  rw [mul_comm, ← Nat.card_prod]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    ⊢ Eq (Nat.card M) (Nat.card (Prod (HasQuotient.Quotient M S) (Subtype fun x => …
  -/
  exact Nat.card_congr AddSubgroup.addGroupEquivQuotientProdAddSubgroup
  /-
    🎉 no goals
  -/


theorem strictMono_comap_prod_map :
    StrictMono fun m : Submodule R M ↦ (m.comap p.subtype, m.map p.mkQ) :=
  fun m₁ m₂ ↦ QuotientAddGroup.strictMono_comap_prod_map
    p.toAddSubgroup (a := m₁.toAddSubgroup) (b := m₂.toAddSubgroup)


/-- The map from the quotient of `M` by a submodule `p` to `M₂` induced by a linear map `f : M → M₂`
vanishing on `p`, as a linear map. -/
def liftQ (f : M →ₛₗ[τ₁₂] M₂) (h : p ≤ ker f) : M ⧸ p →ₛₗ[τ₁₂] M₂ :=
  { QuotientAddGroup.lift p.toAddSubgroup f.toAddMonoidHom h with
                    /-
                      R : Type u_1
                      M : Type u_2
                      r : R
                      x y : M
                      inst✝⁵ : Ring R
                      inst✝⁴ : AddCommGroup M
                      inst✝³ : Module R M
                      p p' : Submodule R M
                      R₂ : Type u_3
                      M₂ : Type u_4
                      inst✝² : Ring R₂
                      inst✝¹ : AddCommGroup M₂
                      inst✝ : Module R₂ M₂
                      τ₁₂ : RingHom R R₂
                      f : LinearMap τ₁₂ M M₂
                      h : LE.le p (LinearMap.ker f)
                      ⊢ ∀ (m : R) (x : HasQuotient.Quotient M p), Eq ({ toFun := (↑__src✝).toFun, ma …
                    -/
    map_smul' := by rintro a ⟨x⟩; exact f.map_smulₛₗ a x }
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem liftQ_apply (f : M →ₛₗ[τ₁₂] M₂) {h} (x : M) : p.liftQ f h (Quotient.mk x) = f x :=
  rfl


@[simp]
                                                                               /-
                                                                                 R : Type u_1
                                                                                 M : Type u_2
                                                                                 inst✝⁵ : Ring R
                                                                                 inst✝⁴ : AddCommGroup M
                                                                                 inst✝³ : Module R M
                                                                                 p : Submodule R M
                                                                                 R₂ : Type u_3
                                                                                 M₂ : Type u_4
                                                                                 inst✝² : Ring R₂
                                                                                 inst✝¹ : AddCommGroup M₂
                                                                                 inst✝ : Module R₂ M₂
                                                                                 τ₁₂ : RingHom R R₂
                                                                                 f : LinearMap τ₁₂ M M₂
                                                                                 h : LE.le p (LinearMap.ker f)
                                                                                 ⊢ Eq ((p.liftQ f h).comp p.mkQ) f
                                                                               -/
theorem liftQ_mkQ (f : M →ₛₗ[τ₁₂] M₂) (h) : (p.liftQ f h).comp p.mkQ = f := by ext; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem pi_liftQ_eq_liftQ_pi {ι : Type*} {N : ι → Type*}
    [∀ i, AddCommGroup (N i)] [∀ i, Module R (N i)]
    (f : (i : ι) → M →ₗ[R] (N i)) {p : Submodule R M} (h : ∀ i, p ≤ ker (f i)) :
    LinearMap.pi (fun i ↦ p.liftQ (f i) (h i)) =
      p.liftQ (LinearMap.pi f) (LinearMap.ker_pi f ▸ le_iInf h) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_5
    N : ι → Type u_6
    inst✝¹ : (i : ι) → AddCommGroup (N i)
    inst✝ : (i : ι) → Module R (N i)
    f : (i : ι) → LinearMap (RingHom.id R) M (N i)
    p : Submodule R M
    h : ∀ (i : ι), LE.le p (LinearMap.ker (f i))
    ⊢ Eq (LinearMap.pi fun i => p.liftQ (f i) ⋯) (p.liftQ (LinearMap.pi f) ⋯)
  -/
  ext x i
  /-
    case h.h.h
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_5
    N : ι → Type u_6
    inst✝¹ : (i : ι) → AddCommGroup (N i)
    inst✝ : (i : ι) → Module R (N i)
    f : (i : ι) → LinearMap (RingHom.id R) M (N i)
    p : Submodule R M
    h : ∀ (i : ι), LE.le p (LinearMap.ker (f i))
    x : M
    i : ι
    ⊢ Eq (((LinearMap.pi fun i => p.liftQ (f i) ⋯).comp p.mkQ) x i) (((p.liftQ (Li …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Special case of `submodule.liftQ` when `p` is the span of `x`. In this case, the condition on
`f` simply becomes vanishing at `x`. -/
def liftQSpanSingleton (x : M) (f : M →ₛₗ[τ₁₂] M₂) (h : f x = 0) : (M ⧸ R ∙ x) →ₛₗ[τ₁₂] M₂ :=
                        /-
                          R : Type u_1
                          M : Type u_2
                          r : R
                          x✝ y : M
                          inst✝⁵ : Ring R
                          inst✝⁴ : AddCommGroup M
                          inst✝³ : Module R M
                          p p' : Submodule R M
                          R₂ : Type u_3
                          M₂ : Type u_4
                          inst✝² : Ring R₂
                          inst✝¹ : AddCommGroup M₂
                          inst✝ : Module R₂ M₂
                          τ₁₂ : RingHom R R₂
                          x : M
                          f : LinearMap τ₁₂ M M₂
                          h : Eq (f x) 0
                          ⊢ LE.le (Submodule.span R (Singleton.singleton x)) (LinearMap.ker f)
                        -/
  (R ∙ x).liftQ f <| by rw [span_singleton_le_iff_mem, LinearMap.mem_ker, h]
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem liftQSpanSingleton_apply (x : M) (f : M →ₛₗ[τ₁₂] M₂) (h : f x = 0) (y : M) :
    liftQSpanSingleton x f h (Quotient.mk y) = f y :=
  rfl


@[simp]
theorem range_mkQ : range p.mkQ = ⊤ :=
                      /-
                        R : Type u_1
                        M : Type u_2
                        inst✝² : Ring R
                        inst✝¹ : AddCommGroup M
                        inst✝ : Module R M
                        p : Submodule R M
                        ⊢ ∀ (x : HasQuotient.Quotient M p), Membership.mem (LinearMap.range p.mkQ) x
                      -/
  eq_top_iff'.2 <| by rintro ⟨x⟩; exact ⟨x, rfl⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
                                      /-
                                        R : Type u_1
                                        M : Type u_2
                                        inst✝² : Ring R
                                        inst✝¹ : AddCommGroup M
                                        inst✝ : Module R M
                                        p : Submodule R M
                                        ⊢ Eq (LinearMap.ker p.mkQ) p
                                      -/
theorem ker_mkQ : ker p.mkQ = p := by ext; simp
                                           /-
                                             🎉 no goals
                                           -/


theorem le_comap_mkQ (p' : Submodule R (M ⧸ p)) : p ≤ comap p.mkQ p' := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    p' : Submodule R (HasQuotient.Quotient M p)
    ⊢ LE.le p (Submodule.comap p.mkQ p')
  -/
  simpa using (comap_mono bot_le : ker p.mkQ ≤ comap p.mkQ p')
  /-
    🎉 no goals
  -/


@[simp]
theorem mkQ_map_self : map p.mkQ p = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    ⊢ Eq (Submodule.map p.mkQ p) Bot.bot
  -/
  rw [eq_bot_iff, map_le_iff_le_comap, comap_bot, ker_mkQ]
  /-
    🎉 no goals
  -/


@[simp]
                                                                  /-
                                                                    R : Type u_1
                                                                    M : Type u_2
                                                                    inst✝² : Ring R
                                                                    inst✝¹ : AddCommGroup M
                                                                    inst✝ : Module R M
                                                                    p p' : Submodule R M
                                                                    ⊢ Eq (Submodule.comap p.mkQ (Submodule.map p.mkQ p')) (Max.max p p')
                                                                  -/
theorem comap_map_mkQ : comap p.mkQ (map p.mkQ p') = p ⊔ p' := by simp [comap_map_eq, sup_comm]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem map_mkQ_eq_top : map p.mkQ p' = ⊤ ↔ p ⊔ p' = ⊤ := by
  -- Porting note: ambiguity of `map_eq_top_iff` is no longer automatically resolved by preferring
  -- the current namespace
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ Iff (Eq (Submodule.map p.mkQ p') Top.top) (Eq (Max.max p p') Top.top)
  -/
  simp only [LinearMap.map_eq_top_iff p.range_mkQ, sup_comm, ker_mkQ]
  /-
    🎉 no goals
  -/


/-- The map from the quotient of `M` by submodule `p` to the quotient of `M₂` by submodule `q` along
`f : M → M₂` is linear. -/
def mapQ (f : M →ₛₗ[τ₁₂] M₂) (h : p ≤ comap f q) : M ⧸ p →ₛₗ[τ₁₂] M₂ ⧸ q :=
                               /-
                                 R : Type u_1
                                 M : Type u_2
                                 r : R
                                 x y : M
                                 inst✝⁵ : Ring R
                                 inst✝⁴ : AddCommGroup M
                                 inst✝³ : Module R M
                                 p p' : Submodule R M
                                 R₂ : Type u_3
                                 M₂ : Type u_4
                                 inst✝² : Ring R₂
                                 inst✝¹ : AddCommGroup M₂
                                 inst✝ : Module R₂ M₂
                                 τ₁₂ : RingHom R R₂
                                 q : Submodule R₂ M₂
                                 f : LinearMap τ₁₂ M M₂
                                 h : LE.le p (Submodule.comap f q)
                                 ⊢ LE.le p (LinearMap.ker (q.mkQ.comp f))
                               -/
  p.liftQ (q.mkQ.comp f) <| by simpa [ker_comp] using h
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem mapQ_apply (f : M →ₛₗ[τ₁₂] M₂) {h} (x : M) :
    mapQ p q f h (Quotient.mk x) = Quotient.mk (f x) :=
  rfl


theorem mapQ_mkQ (f : M →ₛₗ[τ₁₂] M₂) {h} : (mapQ p q f h).comp p.mkQ = q.mkQ.comp f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    p : Submodule R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝² : Ring R₂
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    f : LinearMap τ₁₂ M M₂
    h : LE.le p (Submodule.comap f q)
    ⊢ Eq ((p.mapQ q f h).comp p.mkQ) (q.mkQ.comp f)
  -/
  ext x; rfl
         /-
           🎉 no goals
         -/


@[simp]
                                                              /-
                                                                R : Type u_1
                                                                M : Type u_2
                                                                r : R
                                                                x y : M
                                                                inst✝⁵ : Ring R
                                                                inst✝⁴ : AddCommGroup M
                                                                inst✝³ : Module R M
                                                                p p' : Submodule R M
                                                                R₂ : Type u_3
                                                                M₂ : Type u_4
                                                                inst✝² : Ring R₂
                                                                inst✝¹ : AddCommGroup M₂
                                                                inst✝ : Module R₂ M₂
                                                                τ₁₂ : RingHom R R₂
                                                                q : Submodule R₂ M₂
                                                                ⊢ LE.le p (Submodule.comap 0 q)
                                                              -/
theorem mapQ_zero (h : p ≤ q.comap (0 : M →ₛₗ[τ₁₂] M₂) := (by simp)) :
                                                              /-
                                                                🎉 no goals
                                                              -/
    p.mapQ q (0 : M →ₛₗ[τ₁₂] M₂) h = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    p : Submodule R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝² : Ring R₂
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    h : optParam (LE.le p (Submodule.comap 0 q)) ⋯
    ⊢ Eq (p.mapQ q 0 h) 0
  -/
  ext
  /-
    case h.h
    R : Type u_1
    M : Type u_2
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    p : Submodule R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝² : Ring R₂
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    q : Submodule R₂ M₂
    h : optParam (LE.le p (Submodule.comap 0 q)) ⋯
    x✝ : M
    ⊢ Eq (((p.mapQ q 0 h).comp p.mkQ) x✝) ((LinearMap.comp 0 p.mkQ) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given submodules `p ⊆ M`, `p₂ ⊆ M₂`, `p₃ ⊆ M₃` and maps `f : M → M₂`, `g : M₂ → M₃` inducing
`mapQ f : M ⧸ p → M₂ ⧸ p₂` and `mapQ g : M₂ ⧸ p₂ → M₃ ⧸ p₃` then
`mapQ (g ∘ f) = (mapQ g) ∘ (mapQ f)`. -/
theorem mapQ_comp {R₃ M₃ : Type*} [Ring R₃] [AddCommGroup M₃] [Module R₃ M₃] (p₂ : Submodule R₂ M₂)
    (p₃ : Submodule R₃ M₃) {τ₂₃ : R₂ →+* R₃} {τ₁₃ : R →+* R₃} [RingHomCompTriple τ₁₂ τ₂₃ τ₁₃]
    (f : M →ₛₗ[τ₁₂] M₂) (g : M₂ →ₛₗ[τ₂₃] M₃) (hf : p ≤ p₂.comap f) (hg : p₂ ≤ p₃.comap g)
    (h := hf.trans (comap_mono hg)) :
    p.mapQ p₃ (g.comp f) h = (p₂.mapQ p₃ g hg).comp (p.mapQ p₂ f hf) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    p : Submodule R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝⁶ : Ring R₂
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    R₃ : Type u_5
    M₃ : Type u_6
    inst✝³ : Ring R₃
    inst✝² : AddCommGroup M₃
    inst✝¹ : Module R₃ M₃
    p₂ : Submodule R₂ M₂
    p₃ : Submodule R₃ M₃
    τ₂₃ : RingHom R₂ R₃
    τ₁₃ : RingHom R R₃
    inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
    f : LinearMap τ₁₂ M M₂
    g : LinearMap τ₂₃ M₂ M₃
    hf : LE.le p (Submodule.comap f p₂)
    hg : LE.le p₂ (Submodule.comap g p₃)
    h : optParam (LE.le p (Submodule.comap f (Submodule.comap g p₃))) ⋯
    ⊢ Eq (p.mapQ p₃ (g.comp f) h) ((p₂.mapQ p₃ g hg).comp (p.mapQ p₂ f hf))
  -/
  ext
  /-
    case h.h
    R : Type u_1
    M : Type u_2
    inst✝⁹ : Ring R
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    p : Submodule R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝⁶ : Ring R₂
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    R₃ : Type u_5
    M₃ : Type u_6
    inst✝³ : Ring R₃
    inst✝² : AddCommGroup M₃
    inst✝¹ : Module R₃ M₃
    p₂ : Submodule R₂ M₂
    p₃ : Submodule R₃ M₃
    τ₂₃ : RingHom R₂ R₃
    τ₁₃ : RingHom R R₃
    inst✝ : RingHomCompTriple τ₁₂ τ₂₃ τ₁₃
    f : LinearMap τ₁₂ M M₂
    g : LinearMap τ₂₃ M₂ M₃
    hf : LE.le p (Submodule.comap f p₂)
    hg : LE.le p₂ (Submodule.comap g p₃)
    h : optParam (LE.le p (Submodule.comap f (Submodule.comap g p₃))) ⋯
    x✝ : M
    ⊢ Eq (((p.mapQ p₃ (g.comp f) h).comp p.mkQ) x✝) ((((p₂.mapQ p₃ g hg).comp (p.m …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
                                                     /-
                                                       R : Type u_1
                                                       M : Type u_2
                                                       r : R
                                                       x y : M
                                                       inst✝⁵ : Ring R
                                                       inst✝⁴ : AddCommGroup M
                                                       inst✝³ : Module R M
                                                       p p' : Submodule R M
                                                       R₂ : Type u_3
                                                       M₂ : Type u_4
                                                       inst✝² : Ring R₂
                                                       inst✝¹ : AddCommGroup M₂
                                                       inst✝ : Module R₂ M₂
                                                       τ₁₂ : RingHom R R₂
                                                       q : Submodule R₂ M₂
                                                       ⊢ LE.le p (Submodule.comap LinearMap.id p)
                                                     -/
theorem mapQ_id (h : p ≤ p.comap LinearMap.id := (by rw [comap_id])) :
                                                     /-
                                                       🎉 no goals
                                                     -/
    p.mapQ p LinearMap.id h = LinearMap.id := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    h : optParam (LE.le p (Submodule.comap LinearMap.id p)) ⋯
    ⊢ Eq (p.mapQ p LinearMap.id h) LinearMap.id
  -/
  ext
  /-
    case h.h
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    h : optParam (LE.le p (Submodule.comap LinearMap.id p)) ⋯
    x✝ : M
    ⊢ Eq (((p.mapQ p LinearMap.id h).comp p.mkQ) x✝) ((LinearMap.id.comp p.mkQ) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mapQ_pow {f : M →ₗ[R] M} (h : p ≤ p.comap f) (k : ℕ)
    (h' : p ≤ p.comap (f ^ k) := p.le_comap_pow_of_le_comap h k) :
    p.mapQ p (f ^ k) h' = p.mapQ p f h ^ k := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    h : LE.le p (Submodule.comap f p)
    k : Nat
    h' : optParam (LE.le p (Submodule.comap (HPow.hPow f k) p)) ⋯
    ⊢ Eq (p.mapQ p (HPow.hPow f k) h') (HPow.hPow (p.mapQ p f h) k)
  -/
  induction' k with k ih
    /-
      case zero
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      f : LinearMap (RingHom.id R) M M
      h : LE.le p (Submodule.comap f p)
      h' : optParam (LE.le p (Submodule.comap (HPow.hPow f 0) p)) ⋯
      ⊢ Eq (p.mapQ p (HPow.hPow f 0) h') (HPow.hPow (p.mapQ p f h) 0)
    -/
  · simp [LinearMap.one_eq_id]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      f : LinearMap (RingHom.id R) M M
      h : LE.le p (Submodule.comap f p)
      k : Nat
      ih : ∀ (h' : optParam (LE.le p (Submodule.comap (HPow.hPow f k) p)) ⋯), Eq (p. …
      h' : optParam (LE.le p (Submodule.comap (HPow.hPow f (HAdd.hAdd k 1)) p)) ⋯
      ⊢ Eq (p.mapQ p (HPow.hPow f (HAdd.hAdd k 1)) h') (HPow.hPow (p.mapQ p f h) (HA …
    -/
  · simp only [LinearMap.iterate_succ]
    -- Porting note: why does any of these `optParams` need to be applied? Why didn't `simp` handle
    -- all of this for us?
    convert mapQ_comp p p p f (f ^ k) h (p.le_comap_pow_of_le_comap h k)
      (h.trans (comap_mono <| p.le_comap_pow_of_le_comap h k))
    /-
      case h.e'_3.h.e'_20
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : Submodule R M
      f : LinearMap (RingHom.id R) M M
      h : LE.le p (Submodule.comap f p)
      k : Nat
      ih : ∀ (h' : optParam (LE.le p (Submodule.comap (HPow.hPow f k) p)) ⋯), Eq (p. …
      h' : optParam (LE.le p (Submodule.comap (HPow.hPow f (HAdd.hAdd k 1)) p)) ⋯
      ⊢ Eq (HPow.hPow (p.mapQ p f h) k) (p.mapQ p (HPow.hPow f k) ⋯)
    -/
    exact (ih _).symm
    /-
      🎉 no goals
    -/


theorem comap_liftQ (f : M →ₛₗ[τ₁₂] M₂) (h) : q.comap (p.liftQ f h) = (q.comap f).map (mkQ p) :=
                  /-
                    R : Type u_1
                    M : Type u_2
                    inst✝⁵ : Ring R
                    inst✝⁴ : AddCommGroup M
                    inst✝³ : Module R M
                    p : Submodule R M
                    R₂ : Type u_3
                    M₂ : Type u_4
                    inst✝² : Ring R₂
                    inst✝¹ : AddCommGroup M₂
                    inst✝ : Module R₂ M₂
                    τ₁₂ : RingHom R R₂
                    q : Submodule R₂ M₂
                    f : LinearMap τ₁₂ M M₂
                    h : LE.le p (LinearMap.ker f)
                    ⊢ LE.le (Submodule.comap (p.liftQ f h) q) (Submodule.map p.mkQ (Submodule.coma …
                  -/
  le_antisymm (by rintro ⟨x⟩ hx; exact ⟨_, hx, rfl⟩)
                                 /-
                                   🎉 no goals
                                 -/
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁵ : Ring R
          inst✝⁴ : AddCommGroup M
          inst✝³ : Module R M
          p : Submodule R M
          R₂ : Type u_3
          M₂ : Type u_4
          inst✝² : Ring R₂
          inst✝¹ : AddCommGroup M₂
          inst✝ : Module R₂ M₂
          τ₁₂ : RingHom R R₂
          q : Submodule R₂ M₂
          f : LinearMap τ₁₂ M M₂
          h : LE.le p (LinearMap.ker f)
          ⊢ LE.le (Submodule.map p.mkQ (Submodule.comap f q)) (Submodule.comap (p.liftQ  …
        -/
    (by rw [map_le_iff_le_comap, ← comap_comp, liftQ_mkQ])
        /-
          🎉 no goals
        -/


theorem map_liftQ [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) (h) (q : Submodule R (M ⧸ p)) :
    q.map (p.liftQ f h) = (q.comap p.mkQ).map f :=
                  /-
                    R : Type u_1
                    M : Type u_2
                    inst✝⁶ : Ring R
                    inst✝⁵ : AddCommGroup M
                    inst✝⁴ : Module R M
                    p : Submodule R M
                    R₂ : Type u_3
                    M₂ : Type u_4
                    inst✝³ : Ring R₂
                    inst✝² : AddCommGroup M₂
                    inst✝¹ : Module R₂ M₂
                    τ₁₂ : RingHom R R₂
                    inst✝ : RingHomSurjective τ₁₂
                    f : LinearMap τ₁₂ M M₂
                    h : LE.le p (LinearMap.ker f)
                    q : Submodule R (HasQuotient.Quotient M p)
                    ⊢ LE.le (Submodule.map (p.liftQ f h) q) (Submodule.map f (Submodule.comap p.mk …
                  -/
  le_antisymm (by rintro _ ⟨⟨x⟩, hxq, rfl⟩; exact ⟨x, hxq, rfl⟩)
                                            /-
                                              🎉 no goals
                                            -/
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁶ : Ring R
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          p : Submodule R M
          R₂ : Type u_3
          M₂ : Type u_4
          inst✝³ : Ring R₂
          inst✝² : AddCommGroup M₂
          inst✝¹ : Module R₂ M₂
          τ₁₂ : RingHom R R₂
          inst✝ : RingHomSurjective τ₁₂
          f : LinearMap τ₁₂ M M₂
          h : LE.le p (LinearMap.ker f)
          q : Submodule R (HasQuotient.Quotient M p)
          ⊢ LE.le (Submodule.map f (Submodule.comap p.mkQ q)) (Submodule.map (p.liftQ f  …
        -/
    (by rintro _ ⟨x, hxq, rfl⟩; exact ⟨Quotient.mk x, hxq, rfl⟩)
                                /-
                                  🎉 no goals
                                -/


theorem ker_liftQ (f : M →ₛₗ[τ₁₂] M₂) (h) : ker (p.liftQ f h) = (ker f).map (mkQ p) :=
  comap_liftQ _ _ _ _


theorem range_liftQ [RingHomSurjective τ₁₂] (f : M →ₛₗ[τ₁₂] M₂) (h) :
                                        /-
                                          R : Type u_1
                                          M : Type u_2
                                          inst✝⁶ : Ring R
                                          inst✝⁵ : AddCommGroup M
                                          inst✝⁴ : Module R M
                                          p : Submodule R M
                                          R₂ : Type u_3
                                          M₂ : Type u_4
                                          inst✝³ : Ring R₂
                                          inst✝² : AddCommGroup M₂
                                          inst✝¹ : Module R₂ M₂
                                          τ₁₂ : RingHom R R₂
                                          inst✝ : RingHomSurjective τ₁₂
                                          f : LinearMap τ₁₂ M M₂
                                          h : LE.le p (LinearMap.ker f)
                                          ⊢ Eq (LinearMap.range (p.liftQ f h)) (LinearMap.range f)
                                        -/
    range (p.liftQ f h) = range f := by simpa only [range_eq_map] using map_liftQ _ _ _ _
                                        /-
                                          🎉 no goals
                                        -/


theorem ker_liftQ_eq_bot (f : M →ₛₗ[τ₁₂] M₂) (h) (h' : ker f ≤ p) : ker (p.liftQ f h) = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    p : Submodule R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝² : Ring R₂
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    f : LinearMap τ₁₂ M M₂
    h : LE.le p (LinearMap.ker f)
    h' : LE.le (LinearMap.ker f) p
    ⊢ Eq (LinearMap.ker (p.liftQ f h)) Bot.bot
  -/
  rw [ker_liftQ, le_antisymm h h', mkQ_map_self]
  /-
    🎉 no goals
  -/


theorem ker_liftQ_eq_bot' (f : M →ₛₗ[τ₁₂] M₂) (h : p = ker f) :
    ker (p.liftQ f (le_of_eq h)) = ⊥ :=
  ker_liftQ_eq_bot p f h.le h.ge


/-- The correspondence theorem for modules: there is an order isomorphism between submodules of the
quotient of `M` by `p`, and submodules of `M` larger than `p`. -/
def comapMkQRelIso : Submodule R (M ⧸ p) ≃o { p' : Submodule R M // p ≤ p' } where
  toFun p' := ⟨comap p.mkQ p', le_comap_mkQ p _⟩
  invFun q := map p.mkQ q
                                         /-
                                           R : Type u_1
                                           M : Type u_2
                                           r : R
                                           x y : M
                                           inst✝⁵ : Ring R
                                           inst✝⁴ : AddCommGroup M
                                           inst✝³ : Module R M
                                           p p'✝ : Submodule R M
                                           R₂ : Type u_3
                                           M₂ : Type u_4
                                           inst✝² : Ring R₂
                                           inst✝¹ : AddCommGroup M₂
                                           inst✝ : Module R₂ M₂
                                           τ₁₂ : RingHom R R₂
                                           q : Submodule R₂ M₂
                                           p' : Submodule R (HasQuotient.Quotient M p)
                                           ⊢ LE.le p' (LinearMap.range p.mkQ)
                                         -/
  left_inv p' := map_comap_eq_self <| by simp
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      R : Type u_1
                                                      M : Type u_2
                                                      r : R
                                                      x y : M
                                                      inst✝⁵ : Ring R
                                                      inst✝⁴ : AddCommGroup M
                                                      inst✝³ : Module R M
                                                      p p' : Submodule R M
                                                      R₂ : Type u_3
                                                      M₂ : Type u_4
                                                      inst✝² : Ring R₂
                                                      inst✝¹ : AddCommGroup M₂
                                                      inst✝ : Module R₂ M₂
                                                      τ₁₂ : RingHom R R₂
                                                      q✝ : Submodule R₂ M₂
                                                      x✝ : Subtype fun p' => LE.le p p'
                                                      q : Submodule R M
                                                      hq : LE.le p q
                                                      ⊢ Eq ↑((fun p' => ⟨Submodule.comap p.mkQ p', ⋯⟩) ((fun q => Submodule.map p.mk …
                                                    -/
  right_inv := fun ⟨q, hq⟩ => Subtype.ext_val <| by simpa [comap_map_mkQ p]
                                                    /-
                                                      🎉 no goals
                                                    -/
  map_rel_iff' := comap_le_comap_iff <| range_mkQ _


/-- The ordering on submodules of the quotient of `M` by `p` embeds into the ordering on submodules
of `M`. -/
def comapMkQOrderEmbedding : Submodule R (M ⧸ p) ↪o Submodule R M :=
  (RelIso.toRelEmbedding <| comapMkQRelIso p).trans (Subtype.relEmbedding (· ≤ ·) _)


@[simp]
theorem comapMkQOrderEmbedding_eq (p' : Submodule R (M ⧸ p)) :
    comapMkQOrderEmbedding p p' = comap p.mkQ p' :=
  rfl


theorem span_preimage_eq [RingHomSurjective τ₁₂] {f : M →ₛₗ[τ₁₂] M₂} {s : Set M₂} (h₀ : s.Nonempty)
    (h₁ : s ⊆ range f) : span R (f ⁻¹' s) = (span R₂ s).comap f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝³ : Ring R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    s : Set M₂
    h₀ : s.Nonempty
    h₁ : HasSubset.Subset s ↑(LinearMap.range f)
    ⊢ Eq (Submodule.span R (Set.preimage (⇑f) s)) (Submodule.comap f (Submodule.sp …
  -/
  suffices (span R₂ s).comap f ≤ span R (f ⁻¹' s) by exact le_antisymm (span_preimage_le f s) this
  have hk : ker f ≤ span R (f ⁻¹' s) := by
    let y := Classical.choose h₀
    have hy : y ∈ s := Classical.choose_spec h₀
    rw [ker_le_iff]
    use y, h₁ hy
    rw [← Set.singleton_subset_iff] at hy
    exact Set.Subset.trans subset_span (span_mono (Set.preimage_mono hy))
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝³ : Ring R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    s : Set M₂
    h₀ : s.Nonempty
    h₁ : HasSubset.Subset s ↑(LinearMap.range f)
    hk : LE.le (LinearMap.ker f) (Submodule.span R (Set.preimage (⇑f) s))
    ⊢ LE.le (Submodule.comap f (Submodule.span R₂ s)) (Submodule.span R (Set.preim …
  -/
  rw [← left_eq_sup] at hk
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝³ : Ring R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    s : Set M₂
    h₀ : s.Nonempty
    h₁ : HasSubset.Subset s ↑(LinearMap.range f)
    hk : Eq (Submodule.span R (Set.preimage (⇑f) s)) (Max.max (Submodule.span R (S …
    ⊢ LE.le (Submodule.comap f (Submodule.span R₂ s)) (Submodule.span R (Set.preim …
  -/
  rw [range_coe f] at h₁
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝³ : Ring R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    s : Set M₂
    h₀ : s.Nonempty
    h₁ : HasSubset.Subset s (Set.range ⇑f)
    hk : Eq (Submodule.span R (Set.preimage (⇑f) s)) (Max.max (Submodule.span R (S …
    ⊢ LE.le (Submodule.comap f (Submodule.span R₂ s)) (Submodule.span R (Set.preim …
  -/
  rw [hk, ← LinearMap.map_le_map_iff, map_span, map_comap_eq, Set.image_preimage_eq_of_subset h₁]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝³ : Ring R₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    s : Set M₂
    h₀ : s.Nonempty
    h₁ : HasSubset.Subset s (Set.range ⇑f)
    hk : Eq (Submodule.span R (Set.preimage (⇑f) s)) (Max.max (Submodule.span R (S …
    ⊢ LE.le (Min.min (LinearMap.range f) (Submodule.span R₂ s)) (Submodule.span R₂ …
  -/
  exact inf_le_right
  /-
    🎉 no goals
  -/


/-- If `P` is a submodule of `M` and `Q` a submodule of `N`,
and `f : M ≃ₗ N` maps `P` to `Q`, then `M ⧸ P` is equivalent to `N ⧸ Q`. -/
@[simps]
def Quotient.equiv {N : Type*} [AddCommGroup N] [Module R N] (P : Submodule R M)
    (Q : Submodule R N) (f : M ≃ₗ[R] N) (hf : P.map f = Q) : (M ⧸ P) ≃ₗ[R] N ⧸ Q :=
  { P.mapQ Q (f : M →ₗ[R] N) fun _ hx => hf ▸ Submodule.mem_map_of_mem hx with
    toFun := P.mapQ Q (f : M →ₗ[R] N) fun _ hx => hf ▸ Submodule.mem_map_of_mem hx
    invFun :=
      Q.mapQ P (f.symm : N →ₗ[R] M) fun x hx => by
        /-
          R : Type u_1
          M : Type u_2
          r : R
          x✝ y : M
          inst✝⁷ : Ring R
          inst✝⁶ : AddCommGroup M
          inst✝⁵ : Module R M
          p p' : Submodule R M
          R₂ : Type u_3
          M₂ : Type u_4
          inst✝⁴ : Ring R₂
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R₂ M₂
          τ₁₂ : RingHom R R₂
          q : Submodule R₂ M₂
          N : Type u_5
          inst✝¹ : AddCommGroup N
          inst✝ : Module R N
          P : Submodule R M
          Q : Submodule R N
          f : LinearEquiv (RingHom.id R) M N
          hf : Eq (Submodule.map f P) Q
          x : N
          hx : Membership.mem Q x
          ⊢ Membership.mem (Submodule.comap (↑f.symm) P) x
        -/
        rw [← hf, Submodule.mem_map] at hx
        /-
          R : Type u_1
          M : Type u_2
          r : R
          x✝ y : M
          inst✝⁷ : Ring R
          inst✝⁶ : AddCommGroup M
          inst✝⁵ : Module R M
          p p' : Submodule R M
          R₂ : Type u_3
          M₂ : Type u_4
          inst✝⁴ : Ring R₂
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R₂ M₂
          τ₁₂ : RingHom R R₂
          q : Submodule R₂ M₂
          N : Type u_5
          inst✝¹ : AddCommGroup N
          inst✝ : Module R N
          P : Submodule R M
          Q : Submodule R N
          f : LinearEquiv (RingHom.id R) M N
          hf : Eq (Submodule.map f P) Q
          x : N
          hx : Exists fun y => And (Membership.mem P y) (Eq (f y) x)
          ⊢ Membership.mem (Submodule.comap (↑f.symm) P) x
        -/
        obtain ⟨y, hy, rfl⟩ := hx
        /-
          case intro.intro
          R : Type u_1
          M : Type u_2
          r : R
          x y✝ : M
          inst✝⁷ : Ring R
          inst✝⁶ : AddCommGroup M
          inst✝⁵ : Module R M
          p p' : Submodule R M
          R₂ : Type u_3
          M₂ : Type u_4
          inst✝⁴ : Ring R₂
          inst✝³ : AddCommGroup M₂
          inst✝² : Module R₂ M₂
          τ₁₂ : RingHom R R₂
          q : Submodule R₂ M₂
          N : Type u_5
          inst✝¹ : AddCommGroup N
          inst✝ : Module R N
          P : Submodule R M
          Q : Submodule R N
          f : LinearEquiv (RingHom.id R) M N
          hf : Eq (Submodule.map f P) Q
          y : M
          hy : Membership.mem P y
          ⊢ Membership.mem (Submodule.comap (↑f.symm) P) (f y)
        -/
        simpa
        /-
          🎉 no goals
        -/
                                                                 /-
                                                                   R : Type u_1
                                                                   M : Type u_2
                                                                   r : R
                                                                   x✝ y : M
                                                                   inst✝⁷ : Ring R
                                                                   inst✝⁶ : AddCommGroup M
                                                                   inst✝⁵ : Module R M
                                                                   p p' : Submodule R M
                                                                   R₂ : Type u_3
                                                                   M₂ : Type u_4
                                                                   inst✝⁴ : Ring R₂
                                                                   inst✝³ : AddCommGroup M₂
                                                                   inst✝² : Module R₂ M₂
                                                                   τ₁₂ : RingHom R R₂
                                                                   q : Submodule R₂ M₂
                                                                   N : Type u_5
                                                                   inst✝¹ : AddCommGroup N
                                                                   inst✝ : Module R N
                                                                   P : Submodule R M
                                                                   Q : Submodule R N
                                                                   f : LinearEquiv (RingHom.id R) M N
                                                                   hf : Eq (Submodule.map f P) Q
                                                                   x : HasQuotient.Quotient M P
                                                                   ⊢ ∀ (z : M), Eq ((Q.mapQ P ↑f.symm ⋯) ({ toFun := ⇑(P.mapQ Q ↑f ⋯), map_add' : …
                                                                 -/
    left_inv := fun x => Submodule.Quotient.induction_on _ x (by simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                  /-
                                                                    R : Type u_1
                                                                    M : Type u_2
                                                                    r : R
                                                                    x✝ y : M
                                                                    inst✝⁷ : Ring R
                                                                    inst✝⁶ : AddCommGroup M
                                                                    inst✝⁵ : Module R M
                                                                    p p' : Submodule R M
                                                                    R₂ : Type u_3
                                                                    M₂ : Type u_4
                                                                    inst✝⁴ : Ring R₂
                                                                    inst✝³ : AddCommGroup M₂
                                                                    inst✝² : Module R₂ M₂
                                                                    τ₁₂ : RingHom R R₂
                                                                    q : Submodule R₂ M₂
                                                                    N : Type u_5
                                                                    inst✝¹ : AddCommGroup N
                                                                    inst✝ : Module R N
                                                                    P : Submodule R M
                                                                    Q : Submodule R N
                                                                    f : LinearEquiv (RingHom.id R) M N
                                                                    hf : Eq (Submodule.map f P) Q
                                                                    x : HasQuotient.Quotient N Q
                                                                    ⊢ ∀ (z : N), Eq ({ toFun := ⇑(P.mapQ Q ↑f ⋯), map_add' := ⋯, map_smul' := ⋯ }. …
                                                                  -/
    right_inv := fun x => Submodule.Quotient.induction_on _ x (by simp) }
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem Quotient.equiv_symm {R M N : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    [AddCommGroup N] [Module R N] (P : Submodule R M) (Q : Submodule R N) (f : M ≃ₗ[R] N)
    (hf : P.map f = Q) :
    (Quotient.equiv P Q f hf).symm =
      Quotient.equiv Q P f.symm ((Submodule.map_symm_eq_iff f).mpr hf) :=
  rfl


@[simp]
theorem Quotient.equiv_trans {N O : Type*} [AddCommGroup N] [Module R N] [AddCommGroup O]
    [Module R O] (P : Submodule R M) (Q : Submodule R N) (S : Submodule R O) (e : M ≃ₗ[R] N)
    (f : N ≃ₗ[R] O) (he : P.map e = Q) (hf : Q.map f = S) (hef : P.map (e.trans f) = S) :
    Quotient.equiv P S (e.trans f) hef =
      (Quotient.equiv P Q e he).trans (Quotient.equiv Q S f hf) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_5
    O : Type u_6
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    P : Submodule R M
    Q : Submodule R N
    S : Submodule R O
    e : LinearEquiv (RingHom.id R) M N
    f : LinearEquiv (RingHom.id R) N O
    he : Eq (Submodule.map e P) Q
    hf : Eq (Submodule.map f Q) S
    hef : Eq (Submodule.map (e.trans f) P) S
    ⊢ Eq (Submodule.Quotient.equiv P S (e.trans f) hef) ((Submodule.Quotient.equiv …
  -/
  ext
  -- `simp` can deal with `hef` depending on `e` and `f`
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_5
    O : Type u_6
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    P : Submodule R M
    Q : Submodule R N
    S : Submodule R O
    e : LinearEquiv (RingHom.id R) M N
    f : LinearEquiv (RingHom.id R) N O
    he : Eq (Submodule.map e P) Q
    hf : Eq (Submodule.map f Q) S
    hef : Eq (Submodule.map (e.trans f) P) S
    x✝ : HasQuotient.Quotient M P
    ⊢ Eq ((Submodule.Quotient.equiv P S (e.trans f) hef) x✝) (((Submodule.Quotient …
  -/
  simp only [Quotient.equiv_apply, LinearEquiv.trans_apply, LinearEquiv.coe_trans]
  -- `rw` can deal with `mapQ_comp` needing extra hypotheses coming from the RHS
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_5
    O : Type u_6
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup O
    inst✝ : Module R O
    P : Submodule R M
    Q : Submodule R N
    S : Submodule R O
    e : LinearEquiv (RingHom.id R) M N
    f : LinearEquiv (RingHom.id R) N O
    he : Eq (Submodule.map e P) Q
    hf : Eq (Submodule.map f Q) S
    hef : Eq (Submodule.map (e.trans f) P) S
    x✝ : HasQuotient.Quotient M P
    ⊢ Eq ((P.mapQ S ((↑f).comp ↑e) ⋯) x✝) ((Q.mapQ S ↑f ⋯) ((P.mapQ Q ↑e ⋯) x✝))
  -/
  rw [mapQ_comp, LinearMap.comp_apply]
  /-
    🎉 no goals
  -/


theorem range_mkQ_comp (f : M →ₛₗ[τ₁₂] M₂) : f.range.mkQ.comp f = 0 :=
                            /-
                              R : Type u_1
                              M : Type u_2
                              R₂ : Type u_3
                              M₂ : Type u_4
                              inst✝⁶ : Ring R
                              inst✝⁵ : Ring R₂
                              inst✝⁴ : AddCommMonoid M
                              inst✝³ : AddCommGroup M₂
                              inst✝² : Module R M
                              inst✝¹ : Module R₂ M₂
                              τ₁₂ : RingHom R R₂
                              inst✝ : RingHomSurjective τ₁₂
                              f : LinearMap τ₁₂ M M₂
                              x : M
                              ⊢ Eq (((LinearMap.range f).mkQ.comp f) x) (0 x)
                            -/
  LinearMap.ext fun x => by simp
                            /-
                              🎉 no goals
                            -/


theorem ker_le_range_iff {f : M →ₛₗ[τ₁₂] M₂} {g : M₂ →ₛₗ[τ₂₃] M₃} :
    ker g ≤ range f ↔ f.range.mkQ.comp g.ker.subtype = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    R₂ : Type u_3
    M₂ : Type u_4
    R₃ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : Ring R
    inst✝⁸ : Ring R₂
    inst✝⁷ : Ring R₃
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    inst✝¹ : Module R₃ M₃
    τ₁₂ : RingHom R R₂
    τ₂₃ : RingHom R₂ R₃
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    g : LinearMap τ₂₃ M₂ M₃
    ⊢ Iff (LE.le (LinearMap.ker g) (LinearMap.range f)) (Eq ((LinearMap.range f).m …
  -/
  rw [← range_le_ker_iff, Submodule.ker_mkQ, Submodule.range_subtype]
  /-
    🎉 no goals
  -/


/-- An epimorphism is surjective. -/
theorem range_eq_top_of_cancel {f : M →ₛₗ[τ₁₂] M₂}
    (h : ∀ u v : M₂ →ₗ[R₂] M₂ ⧸ (range f), u.comp f = v.comp f → u = v) : range f = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : Ring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    h : ∀ (u v : LinearMap (RingHom.id R₂) M₂ (HasQuotient.Quotient M₂ (LinearMap. …
    ⊢ Eq (LinearMap.range f) Top.top
  -/
  have h₁ : (0 : M₂ →ₗ[R₂] M₂ ⧸ (range f)).comp f = 0 := zero_comp _
  /-
    R : Type u_1
    M : Type u_2
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : Ring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    h : ∀ (u v : LinearMap (RingHom.id R₂) M₂ (HasQuotient.Quotient M₂ (LinearMap. …
    h₁ : Eq (LinearMap.comp 0 f) 0
    ⊢ Eq (LinearMap.range f) Top.top
  -/
  rw [← Submodule.ker_mkQ (range f), ← h 0 f.range.mkQ (Eq.trans h₁ (range_mkQ_comp _).symm)]
  /-
    R : Type u_1
    M : Type u_2
    R₂ : Type u_3
    M₂ : Type u_4
    inst✝⁶ : Ring R
    inst✝⁵ : Ring R₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    h : ∀ (u v : LinearMap (RingHom.id R₂) M₂ (HasQuotient.Quotient M₂ (LinearMap. …
    h₁ : Eq (LinearMap.comp 0 f) 0
    ⊢ Eq (LinearMap.ker 0) Top.top
  -/
  exact ker_zero
  /-
    🎉 no goals
  -/


/-- If `p = ⊥`, then `M / p ≃ₗ[R] M`. -/
def quotEquivOfEqBot (hp : p = ⊥) : (M ⧸ p) ≃ₗ[R] M :=
  LinearEquiv.ofLinear (p.liftQ id <| hp.symm ▸ bot_le) p.mkQ (liftQ_mkQ _ _ _) <|
    p.quot_hom_ext _ LinearMap.id fun _ => rfl


@[simp]
theorem quotEquivOfEqBot_apply_mk (hp : p = ⊥) (x : M) :
    p.quotEquivOfEqBot hp (Quotient.mk x) = x :=
  rfl


@[simp]
theorem quotEquivOfEqBot_symm_apply (hp : p = ⊥) (x : M) :
    (p.quotEquivOfEqBot hp).symm x = (Quotient.mk x) :=
  rfl


@[simp]
theorem coe_quotEquivOfEqBot_symm (hp : p = ⊥) :
    ((p.quotEquivOfEqBot hp).symm : M →ₗ[R] M ⧸ p) = p.mkQ :=
  rfl


@[simp]
theorem Quotient.equiv_refl (P : Submodule R M) (Q : Submodule R M)
    (hf : P.map (LinearEquiv.refl R M : M →ₗ[R] M) = Q) :
                                                                         /-
                                                                           R : Type u_1
                                                                           M : Type u_2
                                                                           r : R
                                                                           x y : M
                                                                           inst✝² : Ring R
                                                                           inst✝¹ : AddCommGroup M
                                                                           inst✝ : Module R M
                                                                           p p' P Q : Submodule R M
                                                                           hf : Eq (Submodule.map (↑(LinearEquiv.refl R M)) P) Q
                                                                           ⊢ Eq P Q
                                                                         -/
    Quotient.equiv P Q (LinearEquiv.refl R M) hf = quotEquivOfEq _ _ (by simpa using hf) :=
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  rfl


/-- Given modules `M`, `M₂` over a commutative ring, together with submodules `p ⊆ M`, `q ⊆ M₂`,
the natural map $\{f ∈ Hom(M, M₂) | f(p) ⊆ q \} \to Hom(M/p, M₂/q)$ is linear. -/
def mapQLinear : compatibleMaps p q →ₗ[R] M ⧸ p →ₗ[R] M₂ ⧸ q where
  toFun f := mapQ _ _ f.val f.property
  map_add' x y := by
    /-
      R : Type u_1
      M : Type u_2
      M₂ : Type u_3
      r : R
      x✝ y✝ : M
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      x y : Subtype fun x => Membership.mem (p.compatibleMaps q) x
      ⊢ Eq ((fun f => p.mapQ q ↑f ⋯) (HAdd.hAdd x y)) (HAdd.hAdd ((fun f => p.mapQ q …
    -/
    ext
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      M₂ : Type u_3
      r : R
      x✝¹ y✝ : M
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      x y : Subtype fun x => Membership.mem (p.compatibleMaps q) x
      x✝ : M
      ⊢ Eq ((((fun f => p.mapQ q ↑f ⋯) (HAdd.hAdd x y)).comp p.mkQ) x✝) (((HAdd.hAdd …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_smul' c f := by
    /-
      R : Type u_1
      M : Type u_2
      M₂ : Type u_3
      r : R
      x y : M
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      c : R
      f : Subtype fun x => Membership.mem (p.compatibleMaps q) x
      ⊢ Eq ({ toFun := fun f => p.mapQ q ↑f ⋯, map_add' := ⋯ }.toFun (HSMul.hSMul c  …
    -/
    ext
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      M₂ : Type u_3
      r : R
      x y : M
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      p : Submodule R M
      q : Submodule R M₂
      c : R
      f : Subtype fun x => Membership.mem (p.compatibleMaps q) x
      x✝ : M
      ⊢ Eq ((({ toFun := fun f => p.mapQ q ↑f ⋯, map_add' := ⋯ }.toFun (HSMul.hSMul  …
    -/
    rfl
    /-
      🎉 no goals
    -/


