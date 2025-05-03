/-- Lifts `L → K` of `F → K` -/
structure Lifts where
  /-- The domain of a lift. -/
  carrier : IntermediateField F E
  /-- The lifted RingHom, expressed as an AlgHom. -/
  emb : carrier →ₐ[F] K


instance : PartialOrder (Lifts F E K) where
  le L₁ L₂ := ∃ h : L₁.carrier ≤ L₂.carrier, ∀ x, L₂.emb (inclusion h x) = L₁.emb x
                           /-
                             F : Type u_1
                             E : Type u_2
                             K : Type u_3
                             inst✝⁴ : Field F
                             inst✝³ : Field E
                             inst✝² : Field K
                             inst✝¹ : Algebra F E
                             inst✝ : Algebra F K
                             S : Set E
                             L : IntermediateField.Lifts F E K
                             ⊢ ∀ (x : Subtype fun x => Membership.mem L.carrier x), Eq (L.emb ((Intermediat …
                           -/
  le_refl L := ⟨le_rfl, by simp⟩
                           /-
                             🎉 no goals
                           -/
  le_trans L₁ L₂ L₃ := by
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L₁ L₂ L₃ : IntermediateField.Lifts F E K
      ⊢ LE.le L₁ L₂ → LE.le L₂ L₃ → LE.le L₁ L₃
    -/
    rintro ⟨h₁₂, h₁₂'⟩ ⟨h₂₃, h₂₃'⟩
    /-
      case intro.intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L₁ L₂ L₃ : IntermediateField.Lifts F E K
      h₁₂ : LE.le L₁.carrier L₂.carrier
      h₁₂' : ∀ (x : Subtype fun x => Membership.mem L₁.carrier x), Eq (L₂.emb ((Inte …
      h₂₃ : LE.le L₂.carrier L₃.carrier
      h₂₃' : ∀ (x : Subtype fun x => Membership.mem L₂.carrier x), Eq (L₃.emb ((Inte …
      ⊢ LE.le L₁ L₃
    -/
    refine ⟨h₁₂.trans h₂₃, fun _ ↦ ?_⟩
    /-
      case intro.intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L₁ L₂ L₃ : IntermediateField.Lifts F E K
      h₁₂ : LE.le L₁.carrier L₂.carrier
      h₁₂' : ∀ (x : Subtype fun x => Membership.mem L₁.carrier x), Eq (L₂.emb ((Inte …
      h₂₃ : LE.le L₂.carrier L₃.carrier
      h₂₃' : ∀ (x : Subtype fun x => Membership.mem L₂.carrier x), Eq (L₃.emb ((Inte …
      x✝ : Subtype fun x => Membership.mem L₁.carrier x
      ⊢ Eq (L₃.emb ((IntermediateField.inclusion ⋯) x✝)) (L₁.emb x✝)
    -/
    rw [← inclusion_inclusion h₁₂ h₂₃, h₂₃', h₁₂']
    /-
      🎉 no goals
    -/
  le_antisymm := by
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      ⊢ ∀ (a b : IntermediateField.Lifts F E K), LE.le a b → LE.le b a → Eq a b
    -/
    rintro ⟨L₁, e₁⟩ ⟨L₂, e₂⟩ ⟨h₁₂, h₁₂'⟩ ⟨h₂₁, h₂₁'⟩
    /-
      case mk.mk.intro.intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L₁ : IntermediateField F E
      e₁ : AlgHom F (Subtype fun x => Membership.mem L₁ x) K
      L₂ : IntermediateField F E
      e₂ : AlgHom F (Subtype fun x => Membership.mem L₂ x) K
      h₁₂ : LE.le { carrier := L₁, emb := e₁ }.carrier { carrier := L₂, emb := e₂ }. …
      h₁₂' : ∀ (x : Subtype fun x => Membership.mem { carrier := L₁, emb := e₁ }.car …
      h₂₁ : LE.le { carrier := L₂, emb := e₂ }.carrier { carrier := L₁, emb := e₁ }. …
      h₂₁' : ∀ (x : Subtype fun x => Membership.mem { carrier := L₂, emb := e₂ }.car …
      ⊢ Eq { carrier := L₁, emb := e₁ } { carrier := L₂, emb := e₂ }
    -/
    obtain rfl : L₁ = L₂ := h₁₂.antisymm h₂₁
    /-
      case mk.mk.intro.intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L₁ : IntermediateField F E
      e₁ e₂ : AlgHom F (Subtype fun x => Membership.mem L₁ x) K
      h₁₂ : LE.le { carrier := L₁, emb := e₁ }.carrier { carrier := L₁, emb := e₂ }. …
      h₁₂' : ∀ (x : Subtype fun x => Membership.mem { carrier := L₁, emb := e₁ }.car …
      h₂₁ : LE.le { carrier := L₁, emb := e₂ }.carrier { carrier := L₁, emb := e₁ }. …
      h₂₁' : ∀ (x : Subtype fun x => Membership.mem { carrier := L₁, emb := e₂ }.car …
      ⊢ Eq { carrier := L₁, emb := e₁ } { carrier := L₁, emb := e₂ }
    -/
    congr
    /-
      case mk.mk.intro.intro.e_emb
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L₁ : IntermediateField F E
      e₁ e₂ : AlgHom F (Subtype fun x => Membership.mem L₁ x) K
      h₁₂ : LE.le { carrier := L₁, emb := e₁ }.carrier { carrier := L₁, emb := e₂ }. …
      h₁₂' : ∀ (x : Subtype fun x => Membership.mem { carrier := L₁, emb := e₁ }.car …
      h₂₁ : LE.le { carrier := L₁, emb := e₂ }.carrier { carrier := L₁, emb := e₁ }. …
      h₂₁' : ∀ (x : Subtype fun x => Membership.mem { carrier := L₁, emb := e₂ }.car …
      ⊢ Eq e₁ e₂
    -/
    exact AlgHom.ext h₂₁'
    /-
      🎉 no goals
    -/


noncomputable instance : OrderBot (Lifts F E K) where
  bot := ⟨⊥, (Algebra.ofId F K).comp (botEquiv F E)⟩
  bot_le L := ⟨bot_le, fun x ↦ by
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L : IntermediateField.Lifts F E K
      x : Subtype fun x => Membership.mem Bot.bot.carrier x
      ⊢ Eq (L.emb ((IntermediateField.inclusion ⋯) x)) (Bot.bot.emb x)
    -/
    obtain ⟨x, rfl⟩ := (botEquiv F E).symm.surjective x
    /-
      case intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L : IntermediateField.Lifts F E K
      x : F
      ⊢ Eq (L.emb ((IntermediateField.inclusion ⋯) ((IntermediateField.botEquiv F E) …
    -/
    simp_rw [AlgHom.comp_apply, AlgHom.coe_coe, AlgEquiv.apply_symm_apply]
    /-
      case intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L : IntermediateField.Lifts F E K
      x : F
      ⊢ Eq (L.emb ((IntermediateField.inclusion ⋯) ((IntermediateField.botEquiv F E) …
    -/
    exact L.emb.commutes x⟩
    /-
      🎉 no goals
    -/


noncomputable instance : Inhabited (Lifts F E K) :=
  ⟨⊥⟩


theorem le_iff : L₁ ≤ L₂ ↔
    ∃ h : L₁.carrier ≤ L₂.carrier, L₂.emb.comp (inclusion h) = L₁.emb := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    L₁ L₂ : IntermediateField.Lifts F E K
    ⊢ Iff (LE.le L₁ L₂) (Exists fun h => Eq (L₂.emb.comp (IntermediateField.inclus …
  -/
  simp_rw [AlgHom.ext_iff]; rfl
                            /-
                              🎉 no goals
                            -/


theorem eq_iff_le_carrier_eq : L₁ = L₂ ↔ L₁ ≤ L₂ ∧ L₁.carrier = L₂.carrier :=
  ⟨fun eq ↦ ⟨eq.le, congr_arg _ eq⟩, fun ⟨le, eq⟩ ↦ le.antisymm ⟨eq.ge, fun x ↦ (le.2 ⟨x, _⟩).symm⟩⟩


theorem eq_iff : L₁ = L₂ ↔
    ∃ h : L₁.carrier = L₂.carrier, L₂.emb.comp (inclusion h.le) = L₁.emb := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    L₁ L₂ : IntermediateField.Lifts F E K
    ⊢ Iff (Eq L₁ L₂) (Exists fun h => Eq (L₂.emb.comp (IntermediateField.inclusion …
  -/
  rw [eq_iff_le_carrier_eq, le_iff]
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    L₁ L₂ : IntermediateField.Lifts F E K
    ⊢ Iff (And (Exists fun h => Eq (L₂.emb.comp (IntermediateField.inclusion h)) L …
  -/
  exact ⟨fun h ↦ ⟨h.2, h.1.2⟩, fun h ↦ ⟨⟨h.1.le, h.2⟩, h.1⟩⟩
  /-
    🎉 no goals
  -/


theorem lt_iff_le_carrier_ne : L₁ < L₂ ↔ L₁ ≤ L₂ ∧ L₁.carrier ≠ L₂.carrier := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    L₁ L₂ : IntermediateField.Lifts F E K
    ⊢ Iff (LT.lt L₁ L₂) (And (LE.le L₁ L₂) (Ne L₁.carrier L₂.carrier))
  -/
  rw [lt_iff_le_and_ne, and_congr_right]; intro h; simp_rw [Ne, eq_iff_le_carrier_eq, h, true_and]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem lt_iff : L₁ < L₂ ↔
    ∃ h : L₁.carrier < L₂.carrier, L₂.emb.comp (inclusion h.le) = L₁.emb := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    L₁ L₂ : IntermediateField.Lifts F E K
    ⊢ Iff (LT.lt L₁ L₂) (Exists fun h => Eq (L₂.emb.comp (IntermediateField.inclus …
  -/
  rw [lt_iff_le_carrier_ne, le_iff]
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    L₁ L₂ : IntermediateField.Lifts F E K
    ⊢ Iff (And (Exists fun h => Eq (L₂.emb.comp (IntermediateField.inclusion h)) L …
  -/
  exact ⟨fun h ↦ ⟨h.1.1.lt_of_ne h.2, h.1.2⟩, fun h ↦ ⟨⟨h.1.le, h.2⟩, h.1.ne⟩⟩
  /-
    🎉 no goals
  -/


theorem le_of_carrier_le_iSup {ι} {ρ : ι → Lifts F E K} {σ τ : Lifts F E K}
    (hσ : ∀ i, ρ i ≤ σ) (hτ : ∀ i, ρ i ≤ τ) (carrier_le : σ.carrier ≤ ⨆ i, (ρ i).carrier) :
    σ ≤ τ :=
  le_iff.mpr ⟨carrier_le.trans (iSup_le fun i ↦ (hτ i).1), algHom_ext_of_eq_adjoin _
      (carrier_le.antisymm (iSup_le fun i ↦ (hσ i).1)|>.trans <| iSup_eq_adjoin _ _) fun x hx ↦
    have ⟨i, hx⟩ := Set.mem_iUnion.mp hx
    ((hτ i).2 ⟨x, hx⟩).trans ((hσ i).2 ⟨x, hx⟩).symm⟩


/-- `σ : L →ₐ[F] K` is an extendible lift ("extendible pair" in [Isaacs1980]) if for every
intermediate field `M` that is finite-dimensional over `L`, `σ` extends to some `M →ₐ[F] K`.
In our definition we only require `M` to be finitely generated over `L`, which is equivalent
if the ambient field `E` is algebraic over `F` (which is the case in our main application).
We also allow the domain of the extension to be an intermediate field that properly contains `M`,
since one can always restrict the domain to `M`. -/
def IsExtendible (σ : Lifts F E K) : Prop :=
  ∀ S : Finset E, ∃ τ ≥ σ, (S : Set E) ⊆ τ.carrier


/-- The union of a chain of lifts. -/
noncomputable def union : Lifts F E K :=
  let t (i : ↑(insert ⊥ c)) := i.val.carrier
  have hc := hc.insert fun _ _ _ ↦ .inl bot_le
  have dir : Directed (· ≤ ·) t := hc.directedOn.directed_val.mono_comp _ fun _ _ h ↦ h.1
  ⟨iSup t, (Subalgebra.iSupLift (toSubalgebra <| t ·) dir (·.val.emb) (fun i j h ↦
    AlgHom.ext fun x ↦ (hc.total i.2 j.2).elim (fun hij ↦ (hij.snd x).symm) fun hji ↦ by
      erw [AlgHom.comp_apply, ← hji.snd (Subalgebra.inclusion h x),
        inclusion_inclusion, inclusion_self, AlgHom.id_apply x]) _ rfl).comp
      (Subalgebra.equivOfEq _ _ <| toSubalgebra_iSup_of_directed dir)⟩


theorem le_union ⦃σ : Lifts F E K⦄ (hσ : σ ∈ c) : σ ≤ union c hc :=
  have hσ := Set.mem_insert_of_mem ⊥ hσ
  let t (i : ↑(insert ⊥ c)) := i.val.carrier
  ⟨le_iSup t ⟨σ, hσ⟩, fun x ↦ by
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      c : Set (IntermediateField.Lifts F E K)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      σ : IntermediateField.Lifts F E K
      hσ✝ : Membership.mem c σ
      hσ : Membership.mem (Insert.insert Bot.bot c) σ
      t : ↑(Insert.insert Bot.bot c) → IntermediateField F E := fun i => (↑i).carrier
      x : Subtype fun x => Membership.mem σ.carrier x
      ⊢ Eq ((IntermediateField.Lifts.union c hc).emb ((IntermediateField.inclusion ⋯ …
    -/
    dsimp only [union, AlgHom.comp_apply]
    exact Subalgebra.iSupLift_inclusion (K := (toSubalgebra <| t ·))
      (i := ⟨σ, hσ⟩) x (le_iSup (toSubalgebra <| t ·) ⟨σ, hσ⟩)⟩


theorem carrier_union : (union c hc).carrier = ⨆ i : c, i.1.carrier :=
                             /-
                               F : Type u_1
                               E : Type u_2
                               K : Type u_3
                               inst✝⁴ : Field F
                               inst✝³ : Field E
                               inst✝² : Field K
                               inst✝¹ : Algebra F E
                               inst✝ : Algebra F K
                               c : Set (IntermediateField.Lifts F E K)
                               hc : IsChain (fun x1 x2 => LE.le x1 x2) c
                               ⊢ ∀ (i : ↑(Insert.insert Bot.bot c)), LE.le (↑i).carrier (iSup fun i => (↑i).c …
                             -/
  le_antisymm (iSup_le <| by rintro ⟨i, rfl|hi⟩; exacts [bot_le, le_iSup_of_le ⟨i, hi⟩ le_rfl]) <|
                                                 /-
                                                   🎉 no goals
                                                 -/
    iSup_le fun i ↦ le_iSup_of_le ⟨i, .inr i.2⟩ le_rfl


/-- A chain of lifts has an upper bound. -/
theorem exists_upper_bound (c : Set (Lifts F E K)) (hc : IsChain (· ≤ ·) c) :
    ∃ ub, ∀ a ∈ c, a ≤ ub := ⟨_, le_union c hc⟩


theorem union_isExtendible [alg : Algebra.IsAlgebraic F E]
    [Nonempty c] (hext : ∀ σ ∈ c, σ.IsExtendible) :
    (union c hc).IsExtendible := fun S ↦ by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Field K
    inst✝² : Algebra F E
    inst✝¹ : Algebra F K
    c : Set (IntermediateField.Lifts F E K)
    hc : IsChain (fun x1 x2 => LE.le x1 x2) c
    alg : Algebra.IsAlgebraic F E
    inst✝ : Nonempty ↑c
    hext : ∀ (σ : IntermediateField.Lifts F E K), Membership.mem c σ → σ.IsExtendi …
    S : Finset E
    ⊢ Exists fun τ => And (GE.ge τ (IntermediateField.Lifts.union c hc)) (HasSubse …
  -/
  let Ω := adjoin F (S : Set E) →ₐ[F] K
  have ⟨ω, hω⟩ : ∃ ω : Ω, ∀ π : c, ∃ θ ≥ π.1, ⟨_, ω⟩ ≤ θ ∧ θ.carrier = π.1.1 ⊔ adjoin F S := by
    by_contra!; choose π hπ using this
    have := finiteDimensional_adjoin (S := (S : Set E)) fun _ _ ↦ (alg.isIntegral).1 _
    have ⟨π₀, hπ₀⟩ := hc.directed.finite_le π
    have ⟨θ, hθπ, hθ⟩ := hext _ π₀.2 S
    rw [← adjoin_le_iff] at hθ
    let θ₀ := θ.emb.comp (inclusion hθ)
    have := (hπ₀ θ₀).trans hθπ
    exact hπ θ₀ ⟨_, θ.emb.comp <| inclusion <| sup_le this.1 hθ⟩
      ⟨le_sup_left, this.2⟩ ⟨le_sup_right, fun _ ↦ rfl⟩ rfl
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Field K
    inst✝² : Algebra F E
    inst✝¹ : Algebra F K
    c : Set (IntermediateField.Lifts F E K)
    hc : IsChain (fun x1 x2 => LE.le x1 x2) c
    alg : Algebra.IsAlgebraic F E
    inst✝ : Nonempty ↑c
    hext : ∀ (σ : IntermediateField.Lifts F E K), Membership.mem c σ → σ.IsExtendi …
    S : Finset E
    Ω : Type (max u_2 u_3) := AlgHom F (Subtype fun x => Membership.mem (Intermedi …
    ω : Ω
    hω : ∀ (π : ↑c), Exists fun θ => And (GE.ge θ ↑π) (And (LE.le { carrier := Int …
    ⊢ Exists fun τ => And (GE.ge τ (IntermediateField.Lifts.union c hc)) (HasSubse …
  -/
  choose θ ge hθ eq using hω
  have : IsChain (· ≤ ·) (Set.range θ) := by
    simp_rw [← restrictScalars_adjoin_eq_sup, restrictScalars_adjoin] at eq
    rintro _ ⟨π₁, rfl⟩ _ ⟨π₂, rfl⟩ -
    wlog h : π₁ ≤ π₂ generalizing π₁ π₂
    · exact (this _ _ <| (hc.total π₁.2 π₂.2).resolve_left h).symm
    refine .inl (le_iff.mpr ⟨?_, algHom_ext_of_eq_adjoin _ (eq _) ?_⟩)
    · rw [eq, eq]; exact adjoin.mono _ _ _ (Set.union_subset_union_left _ h.1)
    rintro x (hx|hx)
    · change (θ π₂).emb (inclusion (ge π₂).1 <| inclusion h.1 ⟨x, hx⟩) =
        (θ π₁).emb (inclusion (ge π₁).1 ⟨x, hx⟩)
      rw [(ge π₁).2, (ge π₂).2, h.2]
    · change (θ π₂).emb (inclusion (hθ π₂).1 ⟨x, subset_adjoin _ _ hx⟩) =
        (θ π₁).emb (inclusion (hθ π₁).1 ⟨x, subset_adjoin _ _ hx⟩)
      rw [(hθ π₁).2, (hθ π₂).2]
  refine ⟨union _ this, le_of_carrier_le_iSup (fun π ↦ le_union c hc π.2)
    (fun π ↦ (ge π).trans <| le_union _ _ ⟨_, rfl⟩) (carrier_union _ _).le, ?_⟩
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field E
    inst✝³ : Field K
    inst✝² : Algebra F E
    inst✝¹ : Algebra F K
    c : Set (IntermediateField.Lifts F E K)
    hc : IsChain (fun x1 x2 => LE.le x1 x2) c
    alg : Algebra.IsAlgebraic F E
    inst✝ : Nonempty ↑c
    hext : ∀ (σ : IntermediateField.Lifts F E K), Membership.mem c σ → σ.IsExtendi …
    S : Finset E
    Ω : Type (max u_2 u_3) := AlgHom F (Subtype fun x => Membership.mem (Intermedi …
    ω : Ω
    θ : ↑c → IntermediateField.Lifts F E K
    ge : ∀ (π : ↑c), GE.ge (θ π) ↑π
    hθ : ∀ (π : ↑c), LE.le { carrier := IntermediateField.adjoin F ↑S, emb := ω }  …
    eq : ∀ (π : ↑c), Eq (θ π).carrier (Max.max (↑π).carrier (IntermediateField.adj …
    this : IsChain (fun x1 x2 => LE.le x1 x2) (Set.range θ)
    ⊢ HasSubset.Subset ↑S ↑(IntermediateField.Lifts.union (Set.range θ) this).carr …
  -/
  simp_rw [carrier_union, iSup_range', eq]
  exact (subset_adjoin _ _).trans (SetLike.coe_subset_coe.mpr <|
    le_sup_right.trans <| le_iSup_of_le (Classical.arbitrary _) le_rfl)


theorem nonempty_algHom_of_exist_lifts_finset [alg : Algebra.IsAlgebraic F E]
    (h : ∀ S : Finset E, ∃ σ : Lifts F E K, (S : Set E) ⊆ σ.carrier) :
    Nonempty (E →ₐ[F] K) := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    ⊢ Nonempty (AlgHom F E K)
  -/
  have : (⊥ : Lifts F E K).IsExtendible := fun S ↦ have ⟨σ, hσ⟩ := h S; ⟨σ, bot_le, hσ⟩
  have ⟨ϕ, hϕ⟩ := zorn_le₀ {ϕ : Lifts F E K | ϕ.IsExtendible}
    fun c hext hc ↦ (isEmpty_or_nonempty c).elim
      (fun _ ↦ ⟨⊥, this, fun ϕ hϕ ↦ isEmptyElim (⟨ϕ, hϕ⟩ : c)⟩)
      fun _ ↦ ⟨_, union_isExtendible c hc hext, le_union c hc⟩
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    ⊢ Nonempty (AlgHom F E K)
  -/
  suffices ϕ.carrier = ⊤ from ⟨ϕ.emb.comp <| ((equivOfEq this).trans topEquiv).symm⟩
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    ⊢ Eq ϕ.carrier Top.top
  -/
  by_contra!
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this : Ne ϕ.carrier Top.top
    ⊢ False
  -/
  obtain ⟨α, -, hα⟩ := SetLike.exists_of_lt this.lt_top
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    ⊢ False
  -/
  let _ : Algebra ϕ.carrier K := ϕ.emb.toAlgebra
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    x✝ : Algebra (Subtype fun x => Membership.mem ϕ.carrier x) K := ϕ.emb.toAlgebra
    ⊢ False
  -/
  let Λ := ϕ.carrier⟮α⟯ →ₐ[ϕ.carrier] K
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    x✝ : Algebra (Subtype fun x => Membership.mem ϕ.carrier x) K := ϕ.emb.toAlgebra
    Λ : Type (max u_2 u_3) := AlgHom (Subtype fun x => Membership.mem ϕ.carrier x) …
    ⊢ False
  -/
  have := finiteDimensional_adjoin (S := {α}) fun _ _ ↦ ((alg.tower_top ϕ.carrier).isIntegral).1 _
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝¹ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this✝ : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    x✝ : Algebra (Subtype fun x => Membership.mem ϕ.carrier x) K := ϕ.emb.toAlgebra
    Λ : Type (max u_2 u_3) := AlgHom (Subtype fun x => Membership.mem ϕ.carrier x) …
    this : FiniteDimensional (Subtype fun x => Membership.mem ϕ.carrier x) (Subtyp …
    ⊢ False
  -/
  let L (σ : Λ) : Lifts F E K := ⟨ϕ.carrier⟮α⟯.restrictScalars F, σ.restrictScalars F⟩
  have hL (σ : Λ) : ϕ < L σ := lt_iff.mpr
    ⟨by simpa only [L, restrictScalars_adjoin_eq_sup, left_lt_sup, adjoin_simple_le_iff],
      AlgHom.coe_ringHom_injective σ.comp_algebraMap⟩
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝¹ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this✝ : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    x✝ : Algebra (Subtype fun x => Membership.mem ϕ.carrier x) K := ϕ.emb.toAlgebra
    Λ : Type (max u_2 u_3) := AlgHom (Subtype fun x => Membership.mem ϕ.carrier x) …
    this : FiniteDimensional (Subtype fun x => Membership.mem ϕ.carrier x) (Subtyp …
    L : Λ → IntermediateField.Lifts F E K := fun σ => { carrier := IntermediateFie …
    hL : ∀ (σ : Λ), LT.lt ϕ (L σ)
    ⊢ False
  -/
  have ⟨(ϕ_ext : ϕ.IsExtendible), ϕ_max⟩ := maximal_iff_forall_gt.mp hϕ
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝¹ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this✝ : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    x✝ : Algebra (Subtype fun x => Membership.mem ϕ.carrier x) K := ϕ.emb.toAlgebra
    Λ : Type (max u_2 u_3) := AlgHom (Subtype fun x => Membership.mem ϕ.carrier x) …
    this : FiniteDimensional (Subtype fun x => Membership.mem ϕ.carrier x) (Subtyp …
    L : Λ → IntermediateField.Lifts F E K := fun σ => { carrier := IntermediateFie …
    hL : ∀ (σ : Λ), LT.lt ϕ (L σ)
    ϕ_ext : ϕ.IsExtendible
    ϕ_max : ∀ ⦃y : IntermediateField.Lifts F E K⦄, LT.lt ϕ y → Not (Membership.mem …
    ⊢ False
  -/
  simp_rw [Set.mem_setOf, IsExtendible] at ϕ_max; push_neg at ϕ_max
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (S : Finset E), Exists fun σ => HasSubset.Subset ↑S ↑σ.carrier
    this✝¹ : Bot.bot.IsExtendible
    ϕ : IntermediateField.Lifts F E K
    hϕ : Maximal (fun x => Membership.mem (setOf fun ϕ => ϕ.IsExtendible) x) ϕ
    this✝ : Ne ϕ.carrier Top.top
    α : E
    hα : Not (Membership.mem ϕ.carrier α)
    x✝ : Algebra (Subtype fun x => Membership.mem ϕ.carrier x) K := ϕ.emb.toAlgebra
    Λ : Type (max u_2 u_3) := AlgHom (Subtype fun x => Membership.mem ϕ.carrier x) …
    this : FiniteDimensional (Subtype fun x => Membership.mem ϕ.carrier x) (Subtyp …
    L : Λ → IntermediateField.Lifts F E K := fun σ => { carrier := IntermediateFie …
    hL : ∀ (σ : Λ), LT.lt ϕ (L σ)
    ϕ_ext : ϕ.IsExtendible
    ϕ_max : ∀ ⦃y : IntermediateField.Lifts F E K⦄, LT.lt ϕ y → Exists fun S => ∀ ( …
    ⊢ False
  -/
  choose S hS using fun σ : Λ ↦ ϕ_max (hL σ)
  classical
  have ⟨θ, hθϕ, hθ⟩ := ϕ_ext ({α} ∪ Finset.univ.biUnion S)
  simp_rw [Finset.coe_union, Set.union_subset_iff, Finset.coe_singleton, Set.singleton_subset_iff,
    Finset.coe_biUnion, Finset.coe_univ, Set.mem_univ, Set.iUnion_true, Set.iUnion_subset_iff] at hθ
  have : ϕ.carrier⟮α⟯.restrictScalars F ≤ θ.carrier := by
    rw [restrictScalars_adjoin_eq_sup, sup_le_iff, adjoin_simple_le_iff]; exact ⟨hθϕ.1, hθ.1⟩
  exact hS ⟨(θ.emb.comp <| inclusion this).toRingHom, hθϕ.2⟩ θ ⟨this, fun _ ↦ rfl⟩ (hθ.2 _)


/-- Given a lift `x` and an integral element `s : E` over `x.carrier` whose conjugates over
`x.carrier` are all in `K`, we can extend the lift to a lift whose carrier contains `s`. -/
theorem exists_lift_of_splits' (x : Lifts F E K) {s : E} (h1 : IsIntegral x.carrier s)
    (h2 : (minpoly x.carrier s).Splits x.emb.toRingHom) : ∃ y, x ≤ y ∧ s ∈ y.carrier :=
  have I2 := (minpoly.degree_pos h1).ne'
  letI : Algebra x.carrier K := x.emb.toRingHom.toAlgebra
  let carrier := x.carrier⟮s⟯.restrictScalars F
  letI : Algebra x.carrier carrier := x.carrier⟮s⟯.toSubalgebra.algebra
  let φ : carrier →ₐ[x.carrier] K := ((algHomAdjoinIntegralEquiv x.carrier h1).symm
    ⟨rootOfSplits x.emb.toRingHom h2 I2, by
      /-
        F : Type u_1
        E : Type u_2
        K : Type u_3
        inst✝⁴ : Field F
        inst✝³ : Field E
        inst✝² : Field K
        inst✝¹ : Algebra F E
        inst✝ : Algebra F K
        x : IntermediateField.Lifts F E K
        s : E
        h1 : IsIntegral (Subtype fun x_1 => Membership.mem x.carrier x_1) s
        h2 : Polynomial.Splits x.emb.toRingHom (minpoly (Subtype fun x_1 => Membership …
        I2 : Ne (minpoly (Subtype fun x_1 => Membership.mem x.carrier x_1) s).degree 0
        this✝ : Algebra (Subtype fun x_1 => Membership.mem x.carrier x_1) K := x.emb.t …
        carrier : IntermediateField F E := IntermediateField.restrictScalars F (Interm …
        this : Algebra (Subtype fun x_1 => Membership.mem x.carrier x_1) (Subtype fun  …
        ⊢ Membership.mem ((minpoly (Subtype fun x_1 => Membership.mem x.carrier x_1) s …
      -/
      rw [mem_aroots, and_iff_right (minpoly.ne_zero h1)]
      /-
        F : Type u_1
        E : Type u_2
        K : Type u_3
        inst✝⁴ : Field F
        inst✝³ : Field E
        inst✝² : Field K
        inst✝¹ : Algebra F E
        inst✝ : Algebra F K
        x : IntermediateField.Lifts F E K
        s : E
        h1 : IsIntegral (Subtype fun x_1 => Membership.mem x.carrier x_1) s
        h2 : Polynomial.Splits x.emb.toRingHom (minpoly (Subtype fun x_1 => Membership …
        I2 : Ne (minpoly (Subtype fun x_1 => Membership.mem x.carrier x_1) s).degree 0
        this✝ : Algebra (Subtype fun x_1 => Membership.mem x.carrier x_1) K := x.emb.t …
        carrier : IntermediateField F E := IntermediateField.restrictScalars F (Interm …
        this : Algebra (Subtype fun x_1 => Membership.mem x.carrier x_1) (Subtype fun  …
        ⊢ Eq ((Polynomial.aeval (Polynomial.rootOfSplits x.emb.toRingHom h2 I2)) (minp …
      -/
      exact map_rootOfSplits x.emb.toRingHom h2 I2⟩)
      /-
        🎉 no goals
      -/
  ⟨⟨carrier, (@algHomEquivSigma F x.carrier carrier K _ _ _ _ _ _ _ _
      (IsScalarTower.of_algebraMap_eq fun _ ↦ rfl)).symm ⟨x.emb, φ⟩⟩,
    ⟨fun z hz ↦ algebraMap_mem x.carrier⟮s⟯ ⟨z, hz⟩, φ.commutes⟩,
    mem_adjoin_simple_self x.carrier s⟩


/-- Given an integral element `s : E` over `F` whose `F`-conjugates are all in `K`,
any lift can be extended to one whose carrier contains `s`. -/
theorem exists_lift_of_splits (x : Lifts F E K) {s : E} (h1 : IsIntegral F s)
    (h2 : (minpoly F s).Splits (algebraMap F K)) : ∃ y, x ≤ y ∧ s ∈ y.carrier :=
  exists_lift_of_splits' x h1.tower_top <| h1.minpoly_splits_tower_top' <| by
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      x : IntermediateField.Lifts F E K
      s : E
      h1 : IsIntegral F s
      h2 : Polynomial.Splits (algebraMap F K) (minpoly F s)
      ⊢ Polynomial.Splits (x.emb.comp (algebraMap F (Subtype fun x_1 => Membership.m …
    -/
    rwa [← x.emb.comp_algebraMap] at h2
    /-
      🎉 no goals
    -/


private theorem exists_algHom_adjoin_of_splits'' {L : IntermediateField F E}
    (f : L →ₐ[F] K) (hK : ∀ s ∈ S, IsIntegral L s ∧ (minpoly L s).Splits f.toRingHom) :
    ∃ φ : adjoin L S →ₐ[F] K, φ.restrictDomain L = f := by
  obtain ⟨φ, hfφ, hφ⟩ := zorn_le_nonempty_Ici₀ _
    (fun c _ hc _ _ ↦ Lifts.exists_upper_bound c hc) ⟨L, f⟩ le_rfl
  refine ⟨φ.emb.comp (inclusion <| (le_extendScalars_iff hfφ.1 <| adjoin L S).mp <|
    adjoin_le_iff.mpr fun s h ↦ ?_), AlgHom.ext hfφ.2⟩
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    L : IntermediateField F E
    f : AlgHom F (Subtype fun x => Membership.mem L x) K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
    φ : IntermediateField.Lifts F E K
    hfφ : LE.le { carrier := L, emb := f } φ
    hφ : IsMax φ
    s : E
    h : Membership.mem S s
    ⊢ Membership.mem (↑(IntermediateField.extendScalars ⋯)) s
  -/
  letI := (inclusion hfφ.1).toAlgebra
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    L : IntermediateField F E
    f : AlgHom F (Subtype fun x => Membership.mem L x) K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
    φ : IntermediateField.Lifts F E K
    hfφ : LE.le { carrier := L, emb := f } φ
    hφ : IsMax φ
    s : E
    h : Membership.mem S s
    this : Algebra (Subtype fun x => Membership.mem { carrier := L, emb := f }.car …
    ⊢ Membership.mem (↑(IntermediateField.extendScalars ⋯)) s
  -/
  letI : SMul L φ.carrier := Algebra.toSMul
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    L : IntermediateField F E
    f : AlgHom F (Subtype fun x => Membership.mem L x) K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
    φ : IntermediateField.Lifts F E K
    hfφ : LE.le { carrier := L, emb := f } φ
    hφ : IsMax φ
    s : E
    h : Membership.mem S s
    this✝ : Algebra (Subtype fun x => Membership.mem { carrier := L, emb := f }.ca …
    this : SMul (Subtype fun x => Membership.mem L x) (Subtype fun x => Membership …
    ⊢ Membership.mem (↑(IntermediateField.extendScalars ⋯)) s
  -/
  have : IsScalarTower L φ.carrier E := ⟨(smul_assoc · (· : E))⟩
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    L : IntermediateField F E
    f : AlgHom F (Subtype fun x => Membership.mem L x) K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
    φ : IntermediateField.Lifts F E K
    hfφ : LE.le { carrier := L, emb := f } φ
    hφ : IsMax φ
    s : E
    h : Membership.mem S s
    this✝¹ : Algebra (Subtype fun x => Membership.mem { carrier := L, emb := f }.c …
    this✝ : SMul (Subtype fun x => Membership.mem L x) (Subtype fun x => Membershi …
    this : IsScalarTower (Subtype fun x => Membership.mem L x) (Subtype fun x => M …
    ⊢ Membership.mem (↑(IntermediateField.extendScalars ⋯)) s
  -/
  have := φ.exists_lift_of_splits' (hK s h).1.tower_top ((hK s h).1.minpoly_splits_tower_top' ?_)
    /-
      case intro.intro.refine_2
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L : IntermediateField F E
      f : AlgHom F (Subtype fun x => Membership.mem L x) K
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
      φ : IntermediateField.Lifts F E K
      hfφ : LE.le { carrier := L, emb := f } φ
      hφ : IsMax φ
      s : E
      h : Membership.mem S s
      this✝² : Algebra (Subtype fun x => Membership.mem { carrier := L, emb := f }.c …
      this✝¹ : SMul (Subtype fun x => Membership.mem L x) (Subtype fun x => Membersh …
      this✝ : IsScalarTower (Subtype fun x => Membership.mem L x) (Subtype fun x =>  …
      this : Exists fun y => And (LE.le φ y) (Membership.mem y.carrier s)
      ⊢ Membership.mem (↑(IntermediateField.extendScalars ⋯)) s
    -/
  · obtain ⟨y, h1, h2⟩ := this
    /-
      case intro.intro.refine_2.intro.intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L : IntermediateField F E
      f : AlgHom F (Subtype fun x => Membership.mem L x) K
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
      φ : IntermediateField.Lifts F E K
      hfφ : LE.le { carrier := L, emb := f } φ
      hφ : IsMax φ
      s : E
      h : Membership.mem S s
      this✝¹ : Algebra (Subtype fun x => Membership.mem { carrier := L, emb := f }.c …
      this✝ : SMul (Subtype fun x => Membership.mem L x) (Subtype fun x => Membershi …
      this : IsScalarTower (Subtype fun x => Membership.mem L x) (Subtype fun x => M …
      y : IntermediateField.Lifts F E K
      h1 : LE.le φ y
      h2 : Membership.mem y.carrier s
      ⊢ Membership.mem (↑(IntermediateField.extendScalars ⋯)) s
    -/
    exact (hφ h1).1 h2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_1
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      L : IntermediateField F E
      f : AlgHom F (Subtype fun x => Membership.mem L x) K
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral (Subtype fun x => Members …
      φ : IntermediateField.Lifts F E K
      hfφ : LE.le { carrier := L, emb := f } φ
      hφ : IsMax φ
      s : E
      h : Membership.mem S s
      this✝¹ : Algebra (Subtype fun x => Membership.mem { carrier := L, emb := f }.c …
      this✝ : SMul (Subtype fun x => Membership.mem L x) (Subtype fun x => Membershi …
      this : IsScalarTower (Subtype fun x => Membership.mem L x) (Subtype fun x => M …
      ⊢ Polynomial.Splits (φ.emb.comp (algebraMap (Subtype fun x => Membership.mem L …
    -/
  · convert (hK s h).2; ext; apply hfφ.2
                             /-
                               🎉 no goals
                             -/


include hK in
theorem exists_algHom_adjoin_of_splits' :
    ∃ φ : adjoin L S →ₐ[F] K, φ.restrictDomain L = f := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    ⊢ Exists fun φ => Eq (AlgHom.restrictDomain L φ) f
  -/
  let L' := (IsScalarTower.toAlgHom F L E).fieldRange
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
    ⊢ Exists fun φ => Eq (AlgHom.restrictDomain L φ) f
  -/
  let f' : L' →ₐ[F] K := f.comp (AlgEquiv.ofInjectiveField _).symm.toAlgHom
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
    f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
    ⊢ Exists fun φ => Eq (AlgHom.restrictDomain L φ) f
  -/
  have := exists_algHom_adjoin_of_splits'' f' (S := S) fun s hs ↦ ?_
    /-
      case refine_2
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁸ : Field F
      inst✝⁷ : Field E
      inst✝⁶ : Field K
      inst✝⁵ : Algebra F E
      inst✝⁴ : Algebra F K
      S : Set E
      L : Type u_4
      inst✝³ : Field L
      inst✝² : Algebra F L
      inst✝¹ : Algebra L E
      inst✝ : IsScalarTower F L E
      f : AlgHom F L K
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
      L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
      f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
      this : Exists fun φ => Eq (AlgHom.restrictDomain (Subtype fun x => Membership. …
      ⊢ Exists fun φ => Eq (AlgHom.restrictDomain L φ) f
    -/
  · obtain ⟨φ, hφ⟩ := this; refine ⟨φ.comp <|
      inclusion (?_ : (adjoin L S).restrictScalars F ≤ (adjoin L' S).restrictScalars F), ?_⟩
      /-
        case refine_2.intro.refine_1
        F : Type u_1
        E : Type u_2
        K : Type u_3
        inst✝⁸ : Field F
        inst✝⁷ : Field E
        inst✝⁶ : Field K
        inst✝⁵ : Algebra F E
        inst✝⁴ : Algebra F K
        S : Set E
        L : Type u_4
        inst✝³ : Field L
        inst✝² : Algebra F L
        inst✝¹ : Algebra L E
        inst✝ : IsScalarTower F L E
        f : AlgHom F L K
        hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
        L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
        f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
        φ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin (Subty …
        hφ : Eq (AlgHom.restrictDomain (Subtype fun x => Membership.mem L' x) φ) f'
        ⊢ LE.le (IntermediateField.restrictScalars F (IntermediateField.adjoin L S)) ( …
      -/
    · simp_rw [← SetLike.coe_subset_coe, coe_restrictScalars, adjoin_subset_adjoin_iff]
      /-
        case refine_2.intro.refine_1
        F : Type u_1
        E : Type u_2
        K : Type u_3
        inst✝⁸ : Field F
        inst✝⁷ : Field E
        inst✝⁶ : Field K
        inst✝⁵ : Algebra F E
        inst✝⁴ : Algebra F K
        S : Set E
        L : Type u_4
        inst✝³ : Field L
        inst✝² : Algebra F L
        inst✝¹ : Algebra L E
        inst✝ : IsScalarTower F L E
        f : AlgHom F L K
        hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
        L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
        f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
        φ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin (Subty …
        hφ : Eq (AlgHom.restrictDomain (Subtype fun x => Membership.mem L' x) φ) f'
        ⊢ And (HasSubset.Subset (Set.range ⇑(algebraMap L E)) ↑(IntermediateField.adjo …
      -/
      exact ⟨subset_adjoin_of_subset_left S (F := L'.toSubfield) le_rfl, subset_adjoin _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.refine_2
        F : Type u_1
        E : Type u_2
        K : Type u_3
        inst✝⁸ : Field F
        inst✝⁷ : Field E
        inst✝⁶ : Field K
        inst✝⁵ : Algebra F E
        inst✝⁴ : Algebra F K
        S : Set E
        L : Type u_4
        inst✝³ : Field L
        inst✝² : Algebra F L
        inst✝¹ : Algebra L E
        inst✝ : IsScalarTower F L E
        f : AlgHom F L K
        hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
        L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
        f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
        φ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin (Subty …
        hφ : Eq (AlgHom.restrictDomain (Subtype fun x => Membership.mem L' x) φ) f'
        ⊢ Eq (AlgHom.restrictDomain L (φ.comp (IntermediateField.inclusion ⋯))) f
      -/
    · ext x
      /-
        case refine_2.intro.refine_2.H
        F : Type u_1
        E : Type u_2
        K : Type u_3
        inst✝⁸ : Field F
        inst✝⁷ : Field E
        inst✝⁶ : Field K
        inst✝⁵ : Algebra F E
        inst✝⁴ : Algebra F K
        S : Set E
        L : Type u_4
        inst✝³ : Field L
        inst✝² : Algebra F L
        inst✝¹ : Algebra L E
        inst✝ : IsScalarTower F L E
        f : AlgHom F L K
        hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
        L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
        f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
        φ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin (Subty …
        hφ : Eq (AlgHom.restrictDomain (Subtype fun x => Membership.mem L' x) φ) f'
        x : L
        ⊢ Eq ((AlgHom.restrictDomain L (φ.comp (IntermediateField.inclusion ⋯))) x) (f …
      -/
      exact congr($hφ _).trans (congr_arg f <| AlgEquiv.symm_apply_apply _ _)
      /-
        🎉 no goals
      -/
  /-
    case refine_1
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
    f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
    s : E
    hs : Membership.mem S s
    ⊢ And (IsIntegral (Subtype fun x => Membership.mem L' x) s) (Polynomial.Splits …
  -/
  letI : Algebra L L' := (AlgEquiv.ofInjectiveField _).toRingHom.toAlgebra
  /-
    case refine_1
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
    f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
    s : E
    hs : Membership.mem S s
    this : Algebra L (Subtype fun x => Membership.mem L' x) := (AlgEquiv.ofInjecti …
    ⊢ And (IsIntegral (Subtype fun x => Membership.mem L' x) s) (Polynomial.Splits …
  -/
  have : IsScalarTower L L' E := IsScalarTower.of_algebraMap_eq' rfl
  /-
    case refine_1
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
    f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
    s : E
    hs : Membership.mem S s
    this✝ : Algebra L (Subtype fun x => Membership.mem L' x) := (AlgEquiv.ofInject …
    this : IsScalarTower L (Subtype fun x => Membership.mem L' x) E
    ⊢ And (IsIntegral (Subtype fun x => Membership.mem L' x) s) (Polynomial.Splits …
  -/
  refine ⟨(hK s hs).1.tower_top, (hK s hs).1.minpoly_splits_tower_top' ?_⟩
  /-
    case refine_1
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁸ : Field F
    inst✝⁷ : Field E
    inst✝⁶ : Field K
    inst✝⁵ : Algebra F E
    inst✝⁴ : Algebra F K
    S : Set E
    L : Type u_4
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra L E
    inst✝ : IsScalarTower F L E
    f : AlgHom F L K
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral L s) (Polynomial.Splits f …
    L' : IntermediateField F E := (IsScalarTower.toAlgHom F L E).fieldRange
    f' : AlgHom F (Subtype fun x => Membership.mem L' x) K := f.comp ↑(AlgEquiv.of …
    s : E
    hs : Membership.mem S s
    this✝ : Algebra L (Subtype fun x => Membership.mem L' x) := (AlgEquiv.ofInject …
    this : IsScalarTower L (Subtype fun x => Membership.mem L' x) E
    ⊢ Polynomial.Splits (f'.comp (algebraMap L (Subtype fun x => Membership.mem L' …
  -/
  convert (hK s hs).2; ext; exact congr_arg f (AlgEquiv.symm_apply_apply _ _)
                            /-
                              🎉 no goals
                            -/


include hK in
theorem exists_algHom_of_adjoin_splits' (hS : adjoin L S = ⊤) :
    ∃ φ : E →ₐ[F] K, φ.restrictDomain L = f :=
  have ⟨φ, hφ⟩ := exists_algHom_adjoin_of_splits' f hK
  ⟨φ.comp (((equivOfEq hS).trans topEquiv).symm.toAlgHom.restrictScalars F), hφ⟩


theorem exists_algHom_of_splits' (hK : ∀ s : E, IsIntegral L s ∧ (minpoly L s).Splits f.toRingHom) :
    ∃ φ : E →ₐ[F] K, φ.restrictDomain L = f :=
  exists_algHom_of_adjoin_splits' f (fun x _ ↦ hK x) (adjoin_univ L E)


theorem exists_algHom_adjoin_of_splits : ∃ φ : adjoin F S →ₐ[F] K, φ.comp (inclusion hL) = f := by
  obtain ⟨φ, hfφ, hφ⟩ := zorn_le_nonempty_Ici₀ _
    (fun c _ hc _ _ ↦ Lifts.exists_upper_bound c hc) ⟨L, f⟩ le_rfl
  /-
    case intro.intro
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
    L : IntermediateField F E
    f : AlgHom F (Subtype fun x => Membership.mem L x) K
    hL : LE.le L (IntermediateField.adjoin F S)
    φ : IntermediateField.Lifts F E K
    hfφ : LE.le { carrier := L, emb := f } φ
    hφ : IsMax φ
    ⊢ Exists fun φ => Eq (φ.comp (IntermediateField.inclusion hL)) f
  -/
  refine ⟨φ.emb.comp (inclusion <| adjoin_le_iff.mpr fun s hs ↦ ?_), ?_⟩
    /-
      case intro.intro.refine_1
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
      L : IntermediateField F E
      f : AlgHom F (Subtype fun x => Membership.mem L x) K
      hL : LE.le L (IntermediateField.adjoin F S)
      φ : IntermediateField.Lifts F E K
      hfφ : LE.le { carrier := L, emb := f } φ
      hφ : IsMax φ
      s : E
      hs : Membership.mem S s
      ⊢ Membership.mem (↑φ.carrier) s
    -/
  · rcases φ.exists_lift_of_splits (hK s hs).1 (hK s hs).2 with ⟨y, h1, h2⟩
    /-
      case intro.intro.refine_1.intro.intro
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
      L : IntermediateField F E
      f : AlgHom F (Subtype fun x => Membership.mem L x) K
      hL : LE.le L (IntermediateField.adjoin F S)
      φ : IntermediateField.Lifts F E K
      hfφ : LE.le { carrier := L, emb := f } φ
      hφ : IsMax φ
      s : E
      hs : Membership.mem S s
      y : IntermediateField.Lifts F E K
      h1 : LE.le φ y
      h2 : Membership.mem y.carrier s
      ⊢ Membership.mem (↑φ.carrier) s
    -/
    exact (hφ h1).1 h2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      S : Set E
      hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
      L : IntermediateField F E
      f : AlgHom F (Subtype fun x => Membership.mem L x) K
      hL : LE.le L (IntermediateField.adjoin F S)
      φ : IntermediateField.Lifts F E K
      hfφ : LE.le { carrier := L, emb := f } φ
      hφ : IsMax φ
      ⊢ Eq ((φ.emb.comp (IntermediateField.inclusion ⋯)).comp (IntermediateField.inc …
    -/
  · ext; apply hfφ.2
         /-
           🎉 no goals
         -/


theorem nonempty_algHom_adjoin_of_splits : Nonempty (adjoin F S →ₐ[F] K) :=
  have ⟨φ, _⟩ := exists_algHom_adjoin_of_splits hK (⊥ : Lifts F E K).emb bot_le; ⟨φ⟩


include hS in
theorem exists_algHom_of_adjoin_splits : ∃ φ : E →ₐ[F] K, φ.comp L.val = f :=
  have ⟨φ, hφ⟩ := exists_algHom_adjoin_of_splits hK f (hS.symm ▸ le_top)
  ⟨φ.comp ((equivOfEq hS).trans topEquiv).symm.toAlgHom, hφ⟩


include hS in
theorem nonempty_algHom_of_adjoin_splits : Nonempty (E →ₐ[F] K) :=
  have ⟨φ, _⟩ := exists_algHom_of_adjoin_splits hK (⊥ : Lifts F E K).emb hS; ⟨φ⟩


theorem exists_algHom_adjoin_of_splits_of_aeval : ∃ φ : adjoin F S →ₐ[F] K, φ ⟨x, hx⟩ = y := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
    x : E
    y : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    hy : Eq ((Polynomial.aeval y) (minpoly F x)) 0
    ⊢ Exists fun φ => Eq (φ ⟨x, hx⟩) y
  -/
  have := isAlgebraic_adjoin (fun s hs ↦ (hK s hs).1)
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
    x : E
    y : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    hy : Eq ((Polynomial.aeval y) (minpoly F x)) 0
    this : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (IntermediateFie …
    ⊢ Exists fun φ => Eq (φ ⟨x, hx⟩) y
  -/
  have ix : IsAlgebraic F _ := Algebra.IsAlgebraic.isAlgebraic (⟨x, hx⟩ : adjoin F S)
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    S : Set E
    hK : ∀ (s : E), Membership.mem S s → And (IsIntegral F s) (Polynomial.Splits ( …
    x : E
    y : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    hy : Eq ((Polynomial.aeval y) (minpoly F x)) 0
    this : Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (IntermediateFie …
    ix : IsAlgebraic F ⟨x, hx⟩
    ⊢ Exists fun φ => Eq (φ ⟨x, hx⟩) y
  -/
  rw [isAlgebraic_iff_isIntegral, isIntegral_iff] at ix
  obtain ⟨φ, hφ⟩ := exists_algHom_adjoin_of_splits hK ((algHomAdjoinIntegralEquiv F ix).symm
    ⟨y, mem_aroots.mpr ⟨minpoly.ne_zero ix, hy⟩⟩) (adjoin_simple_le_iff.mpr hx)
  exact ⟨φ, (DFunLike.congr_fun hφ <| AdjoinSimple.gen F x).trans <|
    algHomAdjoinIntegralEquiv_symm_apply_gen F ix _⟩


include hS in
theorem exists_algHom_of_adjoin_splits_of_aeval : ∃ φ : E →ₐ[F] K, φ x = y :=
  have ⟨φ, hφ⟩ := exists_algHom_adjoin_of_splits_of_aeval hK (hS ▸ mem_top) hy
  ⟨φ.comp ((equivOfEq hS).trans topEquiv).symm.toAlgHom, hφ⟩



theorem exists_algHom_of_splits : ∃ φ : E →ₐ[F] K, φ.comp L.val = f :=
  exists_algHom_of_adjoin_splits (fun x _ ↦ hK' x) f (adjoin_univ F E)


theorem nonempty_algHom_of_splits : Nonempty (E →ₐ[F] K) :=
  nonempty_algHom_of_adjoin_splits (fun x _ ↦ hK' x) (adjoin_univ F E)


theorem exists_algHom_of_splits_of_aeval (hy : aeval y (minpoly F x) = 0) :
    ∃ φ : E →ₐ[F] K, φ x = y :=
  exists_algHom_of_adjoin_splits_of_aeval (fun x _ ↦ hK' x) (adjoin_univ F E) hy


/-- Let `K/F` be an algebraic extension of fields and `L` a field in which all the minimal
polynomial over `F` of elements of `K` splits. Then, for `x ∈ K`, the images of `x` by the
`F`-algebra morphisms from `K` to `L` are exactly the roots in `L` of the minimal polynomial
of `x` over `F`. -/
theorem Algebra.IsAlgebraic.range_eval_eq_rootSet_minpoly_of_splits {F K : Type*} (L : Type*)
    [Field F] [Field K] [Field L] [Algebra F L] [Algebra F K]
    (hA : ∀ x : K, (minpoly F x).Splits (algebraMap F L))
    [Algebra.IsAlgebraic F K] (x : K) :
    (Set.range fun (ψ : K →ₐ[F] L) => ψ x) = (minpoly F x).rootSet L := by
  /-
    F : Type u_1
    K : Type u_2
    L : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra F K
    hA : ∀ (x : K), Polynomial.Splits (algebraMap F L) (minpoly F x)
    inst✝ : Algebra.IsAlgebraic F K
    x : K
    ⊢ Eq (Set.range fun ψ => ψ x) ((minpoly F x).rootSet L)
  -/
  ext a
  /-
    case h
    F : Type u_1
    K : Type u_2
    L : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra F K
    hA : ∀ (x : K), Polynomial.Splits (algebraMap F L) (minpoly F x)
    inst✝ : Algebra.IsAlgebraic F K
    x : K
    a : L
    ⊢ Iff (Membership.mem (Set.range fun ψ => ψ x) a) (Membership.mem ((minpoly F  …
  -/
  rw [mem_rootSet_of_ne (minpoly.ne_zero (Algebra.IsIntegral.isIntegral x))]
  refine ⟨fun ⟨ψ, hψ⟩ ↦ ?_, fun ha ↦ IntermediateField.exists_algHom_of_splits_of_aeval
    (fun x ↦ ⟨Algebra.IsIntegral.isIntegral x, hA x⟩) ha⟩
  /-
    case h
    F : Type u_1
    K : Type u_2
    L : Type u_3
    inst✝⁵ : Field F
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra F L
    inst✝¹ : Algebra F K
    hA : ∀ (x : K), Polynomial.Splits (algebraMap F L) (minpoly F x)
    inst✝ : Algebra.IsAlgebraic F K
    x : K
    a : L
    x✝ : Membership.mem (Set.range fun ψ => ψ x) a
    ψ : AlgHom F K L
    hψ : Eq ((fun ψ => ψ x) ψ) a
    ⊢ Eq ((Polynomial.aeval a) (minpoly F x)) 0
  -/
  rw [← hψ, Polynomial.aeval_algHom_apply ψ x, minpoly.aeval, map_zero]
  /-
    🎉 no goals
  -/


