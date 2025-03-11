theorem merge' {f g} (hf : Nat.Partrec f) (hg : Nat.Partrec g) :
    ∃ h, Nat.Partrec h ∧
      ∀ a, (∀ x ∈ h a, x ∈ f a ∨ x ∈ g a) ∧ ((h a).Dom ↔ (f a).Dom ∨ (g a).Dom) := by
  /-
    f g : PFun Nat Nat
    hf : Nat.Partrec f
    hg : Nat.Partrec g
    ⊢ Exists fun h => And (Nat.Partrec h) (∀ (a : Nat), And (∀ (x : Nat), Membersh …
  -/
  obtain ⟨cf, rfl⟩ := Code.exists_code.1 hf
  /-
    case intro
    g : PFun Nat Nat
    hg : Nat.Partrec g
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    ⊢ Exists fun h => And (Nat.Partrec h) (∀ (a : Nat), And (∀ (x : Nat), Membersh …
  -/
  obtain ⟨cg, rfl⟩ := Code.exists_code.1 hg
  have : Nat.Partrec fun n => Nat.rfindOpt fun k => cf.evaln k n <|> cg.evaln k n :=
    Partrec.nat_iff.1
      (Partrec.rfindOpt <|
        Primrec.option_orElse.to_comp.comp
          (Code.evaln_prim.to_comp.comp <| (snd.pair (const cf)).pair fst)
          (Code.evaln_prim.to_comp.comp <| (snd.pair (const cg)).pair fst))
  /-
    case intro.intro
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    cg : Nat.Partrec.Code
    hg : Nat.Partrec cg.eval
    this : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partrec …
    ⊢ Exists fun h => And (Nat.Partrec h) (∀ (a : Nat), And (∀ (x : Nat), Membersh …
  -/
  refine ⟨_, this, fun n => ?_⟩
  have : ∀ x ∈ rfindOpt fun k ↦ HOrElse.hOrElse (Code.evaln k cf n) fun _x ↦ Code.evaln k cg n,
      x ∈ Code.eval cf n ∨ x ∈ Code.eval cg n := by
    intro x h
    obtain ⟨k, e⟩ := Nat.rfindOpt_spec h
    revert e
    simp only [Option.mem_def]
    cases' e' : cf.evaln k n with y <;> simp <;> intro e
    · exact Or.inr (Code.evaln_sound e)
    · subst y
      exact Or.inl (Code.evaln_sound e')
  /-
    case intro.intro
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    cg : Nat.Partrec.Code
    hg : Nat.Partrec cg.eval
    this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
    n : Nat
    this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
    ⊢ And (∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
  -/
  refine ⟨this, ⟨fun h => (this _ ⟨h, rfl⟩).imp Exists.fst Exists.fst, ?_⟩⟩
  /-
    case intro.intro
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    cg : Nat.Partrec.Code
    hg : Nat.Partrec cg.eval
    this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
    n : Nat
    this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
    ⊢ Or (cf.eval n).Dom (cg.eval n).Dom → (Nat.rfindOpt fun k => HOrElse.hOrElse  …
  -/
  intro h
  /-
    case intro.intro
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    cg : Nat.Partrec.Code
    hg : Nat.Partrec cg.eval
    this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
    n : Nat
    this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
    h : Or (cf.eval n).Dom (cg.eval n).Dom
    ⊢ (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partrec.Code.evaln k cf n) fun x …
  -/
  rw [Nat.rfindOpt_dom]
  /-
    case intro.intro
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    cg : Nat.Partrec.Code
    hg : Nat.Partrec cg.eval
    this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
    n : Nat
    this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
    h : Or (cf.eval n).Dom (cg.eval n).Dom
    ⊢ Exists fun n_1 => Exists fun a => Membership.mem (HOrElse.hOrElse (Nat.Partr …
  -/
  simp only [dom_iff_mem, Code.evaln_complete, Option.mem_def] at h
  /-
    case intro.intro
    cf : Nat.Partrec.Code
    hf : Nat.Partrec cf.eval
    cg : Nat.Partrec.Code
    hg : Nat.Partrec cg.eval
    this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
    n : Nat
    this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
    h : Or (Exists fun y => Exists fun k => Eq (Nat.Partrec.Code.evaln k cf n) (Op …
    ⊢ Exists fun n_1 => Exists fun a => Membership.mem (HOrElse.hOrElse (Nat.Partr …
  -/
  obtain ⟨x, k, e⟩ | ⟨x, k, e⟩ := h
    /-
      case intro.intro.inl.intro.intro
      cf : Nat.Partrec.Code
      hf : Nat.Partrec cf.eval
      cg : Nat.Partrec.Code
      hg : Nat.Partrec cg.eval
      this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
      n : Nat
      this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
      x k : Nat
      e : Eq (Nat.Partrec.Code.evaln k cf n) (Option.some x)
      ⊢ Exists fun n_1 => Exists fun a => Membership.mem (HOrElse.hOrElse (Nat.Partr …
    -/
  · refine ⟨k, x, ?_⟩
    /-
      case intro.intro.inl.intro.intro
      cf : Nat.Partrec.Code
      hf : Nat.Partrec cf.eval
      cg : Nat.Partrec.Code
      hg : Nat.Partrec cg.eval
      this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
      n : Nat
      this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
      x k : Nat
      e : Eq (Nat.Partrec.Code.evaln k cf n) (Option.some x)
      ⊢ Membership.mem (HOrElse.hOrElse (Nat.Partrec.Code.evaln k cf n) fun x => Nat …
    -/
    simp only [e, Option.some_orElse, Option.mem_def]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.intro.intro
      cf : Nat.Partrec.Code
      hf : Nat.Partrec cf.eval
      cg : Nat.Partrec.Code
      hg : Nat.Partrec cg.eval
      this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
      n : Nat
      this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
      x k : Nat
      e : Eq (Nat.Partrec.Code.evaln k cg n) (Option.some x)
      ⊢ Exists fun n_1 => Exists fun a => Membership.mem (HOrElse.hOrElse (Nat.Partr …
    -/
  · refine ⟨k, ?_⟩
    /-
      case intro.intro.inr.intro.intro
      cf : Nat.Partrec.Code
      hf : Nat.Partrec cf.eval
      cg : Nat.Partrec.Code
      hg : Nat.Partrec cg.eval
      this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
      n : Nat
      this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
      x k : Nat
      e : Eq (Nat.Partrec.Code.evaln k cg n) (Option.some x)
      ⊢ Exists fun a => Membership.mem (HOrElse.hOrElse (Nat.Partrec.Code.evaln k cf …
    -/
    cases' cf.evaln k n with y
      /-
        case intro.intro.inr.intro.intro.none
        cf : Nat.Partrec.Code
        hf : Nat.Partrec cf.eval
        cg : Nat.Partrec.Code
        hg : Nat.Partrec cg.eval
        this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
        n : Nat
        this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
        x k : Nat
        e : Eq (Nat.Partrec.Code.evaln k cg n) (Option.some x)
        ⊢ Exists fun a => Membership.mem (HOrElse.hOrElse Option.none fun x => Nat.Par …
      -/
    · exact ⟨x, by simp only [e, Option.mem_def, Option.none_orElse]⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inr.intro.intro.some
        cf : Nat.Partrec.Code
        hf : Nat.Partrec cf.eval
        cg : Nat.Partrec.Code
        hg : Nat.Partrec cg.eval
        this✝ : Nat.Partrec fun n => Nat.rfindOpt fun k => HOrElse.hOrElse (Nat.Partre …
        n : Nat
        this : ∀ (x : Nat), Membership.mem (Nat.rfindOpt fun k => HOrElse.hOrElse (Nat …
        x k : Nat
        e : Eq (Nat.Partrec.Code.evaln k cg n) (Option.some x)
        y : Nat
        ⊢ Exists fun a => Membership.mem (HOrElse.hOrElse (Option.some y) fun x => Nat …
      -/
    · exact ⟨y, by simp only [Option.some_orElse, Option.mem_def]⟩
      /-
        🎉 no goals
      -/


theorem merge' {f g : α →. σ} (hf : Partrec f) (hg : Partrec g) :
    ∃ k : α →. σ,
      Partrec k ∧ ∀ a, (∀ x ∈ k a, x ∈ f a ∨ x ∈ g a) ∧ ((k a).Dom ↔ (f a).Dom ∨ (g a).Dom) := by
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    ⊢ Exists fun k => And (Partrec k) (∀ (a : α), And (∀ (x : σ), Membership.mem ( …
  -/
  let ⟨k, hk, H⟩ := Nat.Partrec.merge' (bind_decode₂_iff.1 hf) (bind_decode₂_iff.1 hg)
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk : Nat.Partrec k
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Membership.mem …
    ⊢ Exists fun k => And (Partrec k) (∀ (a : α), And (∀ (x : σ), Membership.mem ( …
  -/
  let k' (a : α) := (k (encode a)).bind fun n => (decode (α := σ) n : Part σ)
  refine
    ⟨k', ((nat_iff.2 hk).comp Computable.encode).bind (Computable.decode.ofOption.comp snd).to₂,
      fun a => ?_⟩
  have : ∀ x ∈ k' a, x ∈ f a ∨ x ∈ g a := by
    intro x h'
    simp only [k', exists_prop, mem_coe, mem_bind_iff, Option.mem_def] at h'
    obtain ⟨n, hn, hx⟩ := h'
    have := (H _).1 _ hn
    simp only [decode₂_encode, coe_some, bind_some, mem_map_iff] at this
    obtain ⟨a', ha, rfl⟩ | ⟨a', ha, rfl⟩ := this <;> simp only [encodek, Option.some_inj] at hx <;>
      rw [hx] at ha
    · exact Or.inl ha
    · exact Or.inr ha
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk : Nat.Partrec k
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Membership.mem …
    k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
    a : α
    this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
    ⊢ And (∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
  -/
  refine ⟨this, ⟨fun h => (this _ ⟨h, rfl⟩).imp Exists.fst Exists.fst, ?_⟩⟩
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk : Nat.Partrec k
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Membership.mem …
    k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
    a : α
    this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
    ⊢ Or (f a).Dom (g a).Dom → (k' a).Dom
  -/
  intro h
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk : Nat.Partrec k
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Membership.mem …
    k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
    a : α
    this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
    h : Or (f a).Dom (g a).Dom
    ⊢ (k' a).Dom
  -/
  rw [bind_dom]
  have hk : (k (encode a)).Dom :=
    (H _).2.2 (by simpa only [encodek₂, bind_some, coe_some] using h)
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk✝ : Nat.Partrec k
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Membership.mem …
    k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
    a : α
    this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
    h : Or (f a).Dom (g a).Dom
    hk : (k (Encodable.encode a)).Dom
    ⊢ Exists fun h => (↑(Encodable.decode ((k (Encodable.encode a)).get h))).Dom
  -/
  exists hk
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk✝ : Nat.Partrec k
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Membership.mem …
    k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
    a : α
    this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
    h : Or (f a).Dom (g a).Dom
    hk : (k (Encodable.encode a)).Dom
    ⊢ (↑(Encodable.decode ((k (Encodable.encode a)).get hk))).Dom
  -/
  simp only [exists_prop, mem_map_iff, mem_coe, mem_bind_iff, Option.mem_def] at H
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f g : PFun α σ
    hf : Partrec f
    hg : Partrec g
    k : PFun Nat Nat
    hk✝ : Nat.Partrec k
    k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
    a : α
    this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
    h : Or (f a).Dom (g a).Dom
    hk : (k (Encodable.encode a)).Dom
    H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Exists fun a_2 …
    ⊢ (↑(Encodable.decode ((k (Encodable.encode a)).get hk))).Dom
  -/
  obtain ⟨a', _, y, _, e⟩ | ⟨a', _, y, _, e⟩ := (H _).1 _ ⟨hk, rfl⟩ <;>
    /-
      case inl.intro.intro.intro.intro
      α : Type u_1
      σ : Type u_4
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f g : PFun α σ
      hf : Partrec f
      hg : Partrec g
      k : PFun Nat Nat
      hk✝ : Nat.Partrec k
      k' : α → Part σ := fun a => (k (Encodable.encode a)).bind fun n => ↑(Encodable …
      a : α
      this : ∀ (x : σ), Membership.mem (k' a) x → Or (Membership.mem (f a) x) (Membe …
      h : Or (f a).Dom (g a).Dom
      hk : (k (Encodable.encode a)).Dom
      H : ∀ (a : Nat), And (∀ (x : Nat), Membership.mem (k a) x → Or (Exists fun a_2 …
      a' : α
      left✝¹ : Eq (Encodable.decode₂ α (Encodable.encode a)) (Option.some a')
      y : σ
      left✝ : Membership.mem (f a') y
      e : Eq (Encodable.encode y) ((k (Encodable.encode a)).get hk)
      ⊢ (↑(Encodable.decode ((k (Encodable.encode a)).get hk))).Dom
    -/
    /-
      🎉 no goals
    -/
    simp only [e.symm, encodek, coe_some, some_dom]
    /-
      🎉 no goals
    -/


theorem merge {f g : α →. σ} (hf : Partrec f) (hg : Partrec g)
    (H : ∀ (a), ∀ x ∈ f a, ∀ y ∈ g a, x = y) :
    ∃ k : α →. σ, Partrec k ∧ ∀ a x, x ∈ k a ↔ x ∈ f a ∨ x ∈ g a :=
  let ⟨k, hk, K⟩ := merge' hf hg
  ⟨k, hk, fun a x =>
    ⟨(K _).1 _, fun h => by
      /-
        α : Type u_1
        σ : Type u_4
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f g : PFun α σ
        hf : Partrec f
        hg : Partrec g
        H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
        k : PFun α σ
        hk : Partrec k
        K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
        a : α
        x : σ
        h : Or (Membership.mem (f a) x) (Membership.mem (g a) x)
        ⊢ Membership.mem (k a) x
      -/
      have : (k a).Dom := (K _).2.2 (h.imp Exists.fst Exists.fst)
      /-
        α : Type u_1
        σ : Type u_4
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f g : PFun α σ
        hf : Partrec f
        hg : Partrec g
        H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
        k : PFun α σ
        hk : Partrec k
        K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
        a : α
        x : σ
        h : Or (Membership.mem (f a) x) (Membership.mem (g a) x)
        this : (k a).Dom
        ⊢ Membership.mem (k a) x
      -/
      refine ⟨this, ?_⟩
      /-
        α : Type u_1
        σ : Type u_4
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f g : PFun α σ
        hf : Partrec f
        hg : Partrec g
        H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
        k : PFun α σ
        hk : Partrec k
        K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
        a : α
        x : σ
        h : Or (Membership.mem (f a) x) (Membership.mem (g a) x)
        this : (k a).Dom
        ⊢ Eq ((k a).get this) x
      -/
      cases' h with h h <;> cases' (K _).1 _ ⟨this, rfl⟩ with h' h'
        /-
          case inl.inl
          α : Type u_1
          σ : Type u_4
          inst✝¹ : Primcodable α
          inst✝ : Primcodable σ
          f g : PFun α σ
          hf : Partrec f
          hg : Partrec g
          H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
          k : PFun α σ
          hk : Partrec k
          K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
          a : α
          x : σ
          this : (k a).Dom
          h : Membership.mem (f a) x
          h' : Membership.mem (f a) ((k a).get this)
          ⊢ Eq ((k a).get this) x
        -/
      · exact mem_unique h' h
        /-
          🎉 no goals
        -/
        /-
          case inl.inr
          α : Type u_1
          σ : Type u_4
          inst✝¹ : Primcodable α
          inst✝ : Primcodable σ
          f g : PFun α σ
          hf : Partrec f
          hg : Partrec g
          H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
          k : PFun α σ
          hk : Partrec k
          K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
          a : α
          x : σ
          this : (k a).Dom
          h : Membership.mem (f a) x
          h' : Membership.mem (g a) ((k a).get this)
          ⊢ Eq ((k a).get this) x
        -/
      · exact (H _ _ h _ h').symm
        /-
          🎉 no goals
        -/
        /-
          case inr.inl
          α : Type u_1
          σ : Type u_4
          inst✝¹ : Primcodable α
          inst✝ : Primcodable σ
          f g : PFun α σ
          hf : Partrec f
          hg : Partrec g
          H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
          k : PFun α σ
          hk : Partrec k
          K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
          a : α
          x : σ
          this : (k a).Dom
          h : Membership.mem (g a) x
          h' : Membership.mem (f a) ((k a).get this)
          ⊢ Eq ((k a).get this) x
        -/
      · exact H _ _ h' _ h
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          α : Type u_1
          σ : Type u_4
          inst✝¹ : Primcodable α
          inst✝ : Primcodable σ
          f g : PFun α σ
          hf : Partrec f
          hg : Partrec g
          H : ∀ (a : α) (x : σ), Membership.mem (f a) x → ∀ (y : σ), Membership.mem (g a …
          k : PFun α σ
          hk : Partrec k
          K : ∀ (a : α), And (∀ (x : σ), Membership.mem (k a) x → Or (Membership.mem (f  …
          a : α
          x : σ
          this : (k a).Dom
          h : Membership.mem (g a) x
          h' : Membership.mem (g a) ((k a).get this)
          ⊢ Eq ((k a).get this) x
        -/
      · exact mem_unique h' h⟩⟩
        /-
          🎉 no goals
        -/


theorem cond {c : α → Bool} {f : α →. σ} {g : α →. σ} (hc : Computable c) (hf : Partrec f)
    (hg : Partrec g) : Partrec fun a => cond (c a) (f a) (g a) :=
  let ⟨cf, ef⟩ := exists_code.1 hf
  let ⟨cg, eg⟩ := exists_code.1 hg
  ((eval_part.comp (Computable.cond hc (const cf) (const cg)) Computable.encode).bind
    ((@Computable.decode σ _).comp snd).ofOption.to₂).of_eq
                /-
                  α : Type u_1
                  σ : Type u_4
                  inst✝¹ : Primcodable α
                  inst✝ : Primcodable σ
                  c : α → Bool
                  f g : PFun α σ
                  hc : Computable c
                  hf : Partrec f
                  hg : Partrec g
                  cf : Nat.Partrec.Code
                  ef : Eq cf.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
                  cg : Nat.Partrec.Code
                  eg : Eq cg.eval fun n => (↑(Encodable.decode n)).bind fun a => Part.map Encoda …
                  a : α
                  ⊢ Eq (((_root_.cond (c a) cf cg).eval (Encodable.encode a)).bind fun b => ↑(En …
                -/
                              /-
                                🎉 no goals
                              -/
    fun a => by cases c a <;> simp [ef, eg, encodek]
                              /-
                                🎉 no goals
                              -/


nonrec theorem sum_casesOn {f : α → β ⊕ γ} {g : α → β →. σ} {h : α → γ →. σ} (hf : Computable f)
    (hg : Partrec₂ g) (hh : Partrec₂ h) : @Partrec _ σ _ _ fun a => Sum.casesOn (f a) (g a) (h a) :=
  option_some_iff.1 <|
    (cond (sum_casesOn hf (const true).to₂ (const false).to₂)
          (sum_casesOn_left hf (option_some_iff.2 hg).to₂ (const Option.none).to₂)
          (sum_casesOn_right hf (const Option.none).to₂ (option_some_iff.2 hh).to₂)).of_eq
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    σ : Type u_4
                    inst✝³ : Primcodable α
                    inst✝² : Primcodable β
                    inst✝¹ : Primcodable γ
                    inst✝ : Primcodable σ
                    f : α → Sum β γ
                    g : α → PFun β σ
                    h : α → PFun γ σ
                    hf : Computable f
                    hg : Partrec₂ g
                    hh : Partrec₂ h
                    a : α
                    ⊢ Eq (_root_.cond (Sum.casesOn (f a) (fun b => Bool.true) fun b => Bool.false) …
                  -/
                                /-
                                  🎉 no goals
                                -/
      fun a => by cases f a <;> simp only [Bool.cond_true, Bool.cond_false]
                                /-
                                  🎉 no goals
                                -/


/-- A computable predicate is one whose indicator function is computable. -/
def ComputablePred {α} [Primcodable α] (p : α → Prop) :=
  ∃ _ : DecidablePred p, Computable fun a => decide (p a)


/-- A recursively enumerable predicate is one which is the domain of a computable partial function.
 -/
def RePred {α} [Primcodable α] (p : α → Prop) :=
  Partrec fun a => Part.assert (p a) fun _ => Part.some ()


theorem RePred.of_eq {α} [Primcodable α] {p q : α → Prop} (hp : RePred p) (H : ∀ a, p a ↔ q a) :
    RePred q :=
  (funext fun a => propext (H a) : p = q) ▸ hp


theorem Partrec.dom_re {α β} [Primcodable α] [Primcodable β] {f : α →. β} (h : Partrec f) :
    RePred fun a => (f a).Dom :=
                                                                        /-
                                                                          α : Type u_1
                                                                          β : Type u_2
                                                                          inst✝¹ : Primcodable α
                                                                          inst✝ : Primcodable β
                                                                          f : PFun α β
                                                                          h : Partrec f
                                                                          n : α
                                                                          x✝ : Unit
                                                                          ⊢ Iff (Membership.mem (Part.map (fun b => Unit.unit) (f n)) x✝) (Membership.me …
                                                                        -/
  (h.map (Computable.const ()).to₂).of_eq fun n => Part.ext fun _ => by simp [Part.dom_iff_mem]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem ComputablePred.of_eq {α} [Primcodable α] {p q : α → Prop} (hp : ComputablePred p)
    (H : ∀ a, p a ↔ q a) : ComputablePred q :=
  (funext fun a => propext (H a) : p = q) ▸ hp


theorem computable_iff {p : α → Prop} :
    ComputablePred p ↔ ∃ f : α → Bool, Computable f ∧ p = fun a => (f a : Prop) :=
  ⟨fun ⟨_, h⟩ => ⟨_, h, funext fun _ => propext (Bool.decide_iff _).symm⟩, by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      p : α → Prop
      ⊢ (Exists fun f => And (Computable f) (Eq p fun a => Eq (f a) Bool.true)) → Co …
    -/
    rintro ⟨f, h, rfl⟩; exact ⟨by infer_instance, by simpa using h⟩⟩
                        /-
                          🎉 no goals
                        -/


protected theorem not {p : α → Prop} (hp : ComputablePred p) : ComputablePred fun a => ¬p a := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    p : α → Prop
    hp : ComputablePred p
    ⊢ ComputablePred fun a => Not (p a)
  -/
  obtain ⟨f, hf, rfl⟩ := computable_iff.1 hp
  exact
    ⟨by infer_instance,
      (cond hf (const false) (const true)).of_eq fun n => by
        simp only [Bool.not_eq_true]
        cases f n <;> rfl⟩


/-- The computable functions are closed under if-then-else definitions
with computable predicates. -/
theorem ite {f₁ f₂ : ℕ → ℕ} (hf₁ : Computable f₁) (hf₂ : Computable f₂)
    {c : ℕ → Prop} [DecidablePred c] (hc : ComputablePred c) :
    Computable fun k ↦ if c k then f₁ k else f₂ k := by
  /-
    f₁ f₂ : Nat → Nat
    hf₁ : Computable f₁
    hf₂ : Computable f₂
    c : Nat → Prop
    inst✝ : DecidablePred c
    hc : ComputablePred c
    ⊢ Computable fun k => _root_.ite (c k) (f₁ k) (f₂ k)
  -/
  simp_rw [← Bool.cond_decide]
  /-
    f₁ f₂ : Nat → Nat
    hf₁ : Computable f₁
    hf₂ : Computable f₂
    c : Nat → Prop
    inst✝ : DecidablePred c
    hc : ComputablePred c
    ⊢ Computable fun k => cond (Decidable.decide (c k)) (f₁ k) (f₂ k)
  -/
  obtain ⟨inst, hc⟩ := hc
  /-
    case intro
    f₁ f₂ : Nat → Nat
    hf₁ : Computable f₁
    hf₂ : Computable f₂
    c : Nat → Prop
    inst✝ inst : DecidablePred c
    hc : Computable fun a => Decidable.decide (c a)
    ⊢ Computable fun k => cond (Decidable.decide (c k)) (f₁ k) (f₂ k)
  -/
  convert hc.cond hf₁ hf₂
  /-
    🎉 no goals
  -/


theorem to_re {p : α → Prop} (hp : ComputablePred p) : RePred p := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    p : α → Prop
    hp : ComputablePred p
    ⊢ RePred p
  -/
  obtain ⟨f, hf, rfl⟩ := computable_iff.1 hp
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Primcodable α
    f : α → Bool
    hf : Computable f
    hp : ComputablePred fun a => Eq (f a) Bool.true
    ⊢ RePred fun a => Eq (f a) Bool.true
  -/
  unfold RePred
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Primcodable α
    f : α → Bool
    hf : Computable f
    hp : ComputablePred fun a => Eq (f a) Bool.true
    ⊢ Partrec fun a => Part.assert ((fun a => Eq (f a) Bool.true) a) fun x => Part …
  -/
  dsimp only []
  refine
    (Partrec.cond hf (Decidable.Partrec.const' (Part.some ())) Partrec.none).of_eq fun n =>
      Part.ext fun a => ?_
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Primcodable α
    f : α → Bool
    hf : Computable f
    hp : ComputablePred fun a => Eq (f a) Bool.true
    n : α
    a : Unit
    ⊢ Iff (Membership.mem (cond (f n) (Part.some Unit.unit) Part.none) a) (Members …
  -/
                         /-
                           🎉 no goals
                         -/
  cases a; cases f n <;> simp
                         /-
                           🎉 no goals
                         -/


/-- **Rice's Theorem** -/
theorem rice (C : Set (ℕ →. ℕ)) (h : ComputablePred fun c => eval c ∈ C) {f g} (hf : Nat.Partrec f)
    (hg : Nat.Partrec g) (fC : f ∈ C) : g ∈ C := by
  /-
    C : Set (PFun Nat Nat)
    h : ComputablePred fun c => Membership.mem C c.eval
    f g : PFun Nat Nat
    hf : Nat.Partrec f
    hg : Nat.Partrec g
    fC : Membership.mem C f
    ⊢ Membership.mem C g
  -/
  cases' h with _ h
  obtain ⟨c, e⟩ :=
    fixed_point₂
      (Partrec.cond (h.comp fst) ((Partrec.nat_iff.2 hg).comp snd).to₂
          ((Partrec.nat_iff.2 hf).comp snd).to₂).to₂
  /-
    case intro.intro
    C : Set (PFun Nat Nat)
    f g : PFun Nat Nat
    hf : Nat.Partrec f
    hg : Nat.Partrec g
    fC : Membership.mem C f
    w✝ : DecidablePred fun c => Membership.mem C c.eval
    h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
    c : Nat.Partrec.Code
    e : Eq c.eval fun b => cond (Decidable.decide ((fun c => Membership.mem C c.ev …
    ⊢ Membership.mem C g
  -/
  simp only [Bool.cond_decide] at e
  /-
    case intro.intro
    C : Set (PFun Nat Nat)
    f g : PFun Nat Nat
    hf : Nat.Partrec f
    hg : Nat.Partrec g
    fC : Membership.mem C f
    w✝ : DecidablePred fun c => Membership.mem C c.eval
    h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
    c : Nat.Partrec.Code
    e : Eq c.eval fun b => _root_.ite (Membership.mem C c.eval) (g b) (f b)
    ⊢ Membership.mem C g
  -/
  by_cases H : eval c ∈ C
    /-
      case pos
      C : Set (PFun Nat Nat)
      f g : PFun Nat Nat
      hf : Nat.Partrec f
      hg : Nat.Partrec g
      fC : Membership.mem C f
      w✝ : DecidablePred fun c => Membership.mem C c.eval
      h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
      c : Nat.Partrec.Code
      e : Eq c.eval fun b => _root_.ite (Membership.mem C c.eval) (g b) (f b)
      H : Membership.mem C c.eval
      ⊢ Membership.mem C g
    -/
  · simp only [H, if_true] at e
    /-
      case pos
      C : Set (PFun Nat Nat)
      f g : PFun Nat Nat
      hf : Nat.Partrec f
      hg : Nat.Partrec g
      fC : Membership.mem C f
      w✝ : DecidablePred fun c => Membership.mem C c.eval
      h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
      c : Nat.Partrec.Code
      H : Membership.mem C c.eval
      e : Eq c.eval fun b => g b
      ⊢ Membership.mem C g
    -/
    change (fun b => g b) ∈ C
    /-
      case pos
      C : Set (PFun Nat Nat)
      f g : PFun Nat Nat
      hf : Nat.Partrec f
      hg : Nat.Partrec g
      fC : Membership.mem C f
      w✝ : DecidablePred fun c => Membership.mem C c.eval
      h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
      c : Nat.Partrec.Code
      H : Membership.mem C c.eval
      e : Eq c.eval fun b => g b
      ⊢ Membership.mem C fun b => g b
    -/
    rwa [← e]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Set (PFun Nat Nat)
      f g : PFun Nat Nat
      hf : Nat.Partrec f
      hg : Nat.Partrec g
      fC : Membership.mem C f
      w✝ : DecidablePred fun c => Membership.mem C c.eval
      h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
      c : Nat.Partrec.Code
      e : Eq c.eval fun b => _root_.ite (Membership.mem C c.eval) (g b) (f b)
      H : Not (Membership.mem C c.eval)
      ⊢ Membership.mem C g
    -/
  · simp only [H, if_false] at e
    /-
      case neg
      C : Set (PFun Nat Nat)
      f g : PFun Nat Nat
      hf : Nat.Partrec f
      hg : Nat.Partrec g
      fC : Membership.mem C f
      w✝ : DecidablePred fun c => Membership.mem C c.eval
      h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
      c : Nat.Partrec.Code
      H : Not (Membership.mem C c.eval)
      e : Eq c.eval fun b => f b
      ⊢ Membership.mem C g
    -/
    rw [e] at H
    /-
      case neg
      C : Set (PFun Nat Nat)
      f g : PFun Nat Nat
      hf : Nat.Partrec f
      hg : Nat.Partrec g
      fC : Membership.mem C f
      w✝ : DecidablePred fun c => Membership.mem C c.eval
      h : Computable fun a => Decidable.decide ((fun c => Membership.mem C c.eval) a)
      c : Nat.Partrec.Code
      H : Not (Membership.mem C fun b => f b)
      e : Eq c.eval fun b => f b
      ⊢ Membership.mem C g
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem rice₂ (C : Set Code) (H : ∀ cf cg, eval cf = eval cg → (cf ∈ C ↔ cg ∈ C)) :
    (ComputablePred fun c => c ∈ C) ↔ C = ∅ ∨ C = Set.univ := by
  classical exact
      have hC : ∀ f, f ∈ C ↔ eval f ∈ eval '' C := fun f =>
        ⟨Set.mem_image_of_mem _, fun ⟨g, hg, e⟩ => (H _ _ e).1 hg⟩
      ⟨fun h =>
        or_iff_not_imp_left.2 fun C0 =>
          Set.eq_univ_of_forall fun cg =>
            let ⟨cf, fC⟩ := Set.nonempty_iff_ne_empty.2 C0
            (hC _).2 <|
              rice (eval '' C) (h.of_eq hC)
                (Partrec.nat_iff.1 <| eval_part.comp (const cf) Computable.id)
                (Partrec.nat_iff.1 <| eval_part.comp (const cg) Computable.id) ((hC _).1 fC),
        fun h => by {
          obtain rfl | rfl := h <;> simpa [ComputablePred, Set.mem_empty_iff_false] using
            Computable.const _}⟩


/-- The Halting problem is recursively enumerable -/
theorem halting_problem_re (n) : RePred fun c => (eval c n).Dom :=
  (eval_part.comp Computable.id (Computable.const _)).dom_re


/-- The **Halting problem** is not computable -/
theorem halting_problem (n) : ¬ComputablePred fun c => (eval c n).Dom
  | h => rice { f | (f n).Dom } h Nat.Partrec.zero Nat.Partrec.none trivial

-- Post's theorem on the equivalence of r.e., co-r.e. sets and
-- computable sets. The assumption that p is decidable is required
-- unless we assume Markov's principle or LEM.
-- @[nolint decidable_classical]

theorem computable_iff_re_compl_re {p : α → Prop} [DecidablePred p] :
    ComputablePred p ↔ RePred p ∧ RePred fun a => ¬p a :=
  ⟨fun h => ⟨h.to_re, h.not.to_re⟩, fun ⟨h₁, h₂⟩ =>
    ⟨‹_›, by
      obtain ⟨k, pk, hk⟩ :=
        Partrec.merge (h₁.map (Computable.const true).to₂) (h₂.map (Computable.const false).to₂)
        (by
          intro a x hx y hy
          simp only [Part.mem_map_iff, Part.mem_assert_iff, Part.mem_some_iff, exists_prop,
            and_true, exists_const] at hx hy
          cases hy.1 hx.1)
      /-
        case intro.intro
        α : Type u_1
        inst✝¹ : Primcodable α
        p : α → Prop
        inst✝ : DecidablePred p
        x✝ : And (RePred p) (RePred fun a => Not (p a))
        h₁ : RePred p
        h₂ : RePred fun a => Not (p a)
        k : PFun α Bool
        pk : Partrec k
        hk : ∀ (a : α) (x : Bool), Iff (Membership.mem (k a) x) (Or (Membership.mem (P …
        ⊢ Computable fun a => Decidable.decide (p a)
      -/
      refine Partrec.of_eq pk fun n => Part.eq_some_iff.2 ?_
      /-
        case intro.intro
        α : Type u_1
        inst✝¹ : Primcodable α
        p : α → Prop
        inst✝ : DecidablePred p
        x✝ : And (RePred p) (RePred fun a => Not (p a))
        h₁ : RePred p
        h₂ : RePred fun a => Not (p a)
        k : PFun α Bool
        pk : Partrec k
        hk : ∀ (a : α) (x : Bool), Iff (Membership.mem (k a) x) (Or (Membership.mem (P …
        n : α
        ⊢ Membership.mem (k n) ((fun a => Decidable.decide (p a)) n)
      -/
      rw [hk]
      simp only [Part.mem_map_iff, Part.mem_assert_iff, Part.mem_some_iff, exists_prop, and_true,
        true_eq_decide_iff, and_self, exists_const, false_eq_decide_iff]
      /-
        case intro.intro
        α : Type u_1
        inst✝¹ : Primcodable α
        p : α → Prop
        inst✝ : DecidablePred p
        x✝ : And (RePred p) (RePred fun a => Not (p a))
        h₁ : RePred p
        h₂ : RePred fun a => Not (p a)
        k : PFun α Bool
        pk : Partrec k
        hk : ∀ (a : α) (x : Bool), Iff (Membership.mem (k a) x) (Or (Membership.mem (P …
        n : α
        ⊢ Or (p n) (Not (p n))
      -/
      apply Decidable.em⟩⟩
      /-
        🎉 no goals
      -/


theorem computable_iff_re_compl_re' {p : α → Prop} :
    ComputablePred p ↔ RePred p ∧ RePred fun a => ¬p a := by
  /-
    α : Type u_1
    inst✝ : Primcodable α
    p : α → Prop
    ⊢ Iff (ComputablePred p) (And (RePred p) (RePred fun a => Not (p a)))
  -/
  classical exact computable_iff_re_compl_re
  /-
    🎉 no goals
  -/


theorem halting_problem_not_re (n) : ¬RePred fun c => ¬(eval c n).Dom
  | h => halting_problem _ <| computable_iff_re_compl_re'.2 ⟨halting_problem_re _, h⟩


/-- A simplified basis for `Partrec`. -/
inductive Partrec' : ∀ {n}, (List.Vector ℕ n →. ℕ) → Prop
  | prim {n f} : @Primrec' n f → @Partrec' n f
  | comp {m n f} (g : Fin n → List.Vector ℕ m →. ℕ) :
    Partrec' f → (∀ i, Partrec' (g i)) →
      Partrec' fun v => (List.Vector.mOfFn fun i => g i v) >>= f
  | rfind {n} {f : List.Vector ℕ (n + 1) → ℕ} :
    @Partrec' (n + 1) f → Partrec' fun v => rfind fun n => some (f (n ::ᵥ v) = 0)


theorem to_part {n f} (pf : @Partrec' n f) : _root_.Partrec f := by
  induction pf with
  | prim hf => exact hf.to_prim.to_comp
  | comp _ _ _ hf hg => exact (Partrec.vector_mOfFn hg).bind (hf.comp snd)
  | rfind _ hf =>
    have := hf.comp (vector_cons.comp snd fst)
    have :=
      ((Primrec.eq.comp _root_.Primrec.id (_root_.Primrec.const 0)).to_comp.comp
        this).to₂.partrec₂
    exact _root_.Partrec.rfind this


theorem of_eq {n} {f g : List.Vector ℕ n →. ℕ} (hf : Partrec' f) (H : ∀ i, f i = g i) :
    Partrec' g :=
  (funext H : f = g) ▸ hf


theorem of_prim {n} {f : List.Vector ℕ n → ℕ} (hf : Primrec f) : @Partrec' n f :=
  prim (Nat.Primrec'.of_prim hf)


theorem head {n : ℕ} : @Partrec' n.succ (@head ℕ n) :=
  prim Nat.Primrec'.head


theorem tail {n f} (hf : @Partrec' n f) : @Partrec' n.succ fun v => f v.tail :=
  (hf.comp _ fun i => @prim _ _ <| Nat.Primrec'.get i.succ).of_eq fun v => by
    /-
      n : Nat
      f : PFun (List.Vector Nat n) Nat
      hf : Nat.Partrec' f
      v : List.Vector Nat (HAdd.hAdd n 1)
      ⊢ Eq (Bind.bind (List.Vector.mOfFn fun i => (↑fun v => v.get i.succ) v) f) (f  …
    -/
    simp; rw [← ofFn_get v.tail]; congr; funext i; simp
                                                   /-
                                                     🎉 no goals
                                                   -/


protected theorem bind {n f g} (hf : @Partrec' n f) (hg : @Partrec' (n + 1) g) :
    @Partrec' n fun v => (f v).bind fun a => g (a ::ᵥ v) :=
  (@comp n (n + 1) g (fun i => Fin.cases f (fun i v => some (v.get i)) i) hg fun i => by
      /-
        n : Nat
        f : PFun (List.Vector Nat n) Nat
        g : PFun (List.Vector Nat (HAdd.hAdd n 1)) Nat
        hf : Nat.Partrec' f
        hg : Nat.Partrec' g
        i : Fin (HAdd.hAdd n 1)
        ⊢ Nat.Partrec' ((fun i => Fin.cases f (fun i v => ↑(Option.some (v.get i))) i) …
      -/
                                              /-
                                                🎉 no goals
                                              -/
      refine Fin.cases ?_ (fun i => ?_) i <;> simp [*]
      /-
        case refine_2
        n : Nat
        f : PFun (List.Vector Nat n) Nat
        g : PFun (List.Vector Nat (HAdd.hAdd n 1)) Nat
        hf : Nat.Partrec' f
        hg : Nat.Partrec' g
        i✝ : Fin (HAdd.hAdd n 1)
        i : Fin n
        ⊢ Nat.Partrec' fun v => Part.some (v.get i)
      -/
      exact prim (Nat.Primrec'.get _)).of_eq
      /-
        🎉 no goals
      -/
                /-
                  n : Nat
                  f : PFun (List.Vector Nat n) Nat
                  g : PFun (List.Vector Nat (HAdd.hAdd n 1)) Nat
                  hf : Nat.Partrec' f
                  hg : Nat.Partrec' g
                  v : List.Vector Nat n
                  ⊢ Eq (Bind.bind (List.Vector.mOfFn fun i => Fin.cases f (fun i v => ↑(Option.s …
                -/
    fun v => by simp [mOfFn, Part.bind_assoc, pure]
                /-
                  🎉 no goals
                -/


protected theorem map {n f} {g : List.Vector ℕ (n + 1) → ℕ} (hf : @Partrec' n f)
    (hg : @Partrec' (n + 1) g) : @Partrec' n fun v => (f v).map fun a => g (a ::ᵥ v) := by
  /-
    n : Nat
    f : PFun (List.Vector Nat n) Nat
    g : List.Vector Nat (HAdd.hAdd n 1) → Nat
    hf : Nat.Partrec' f
    hg : Nat.Partrec' ↑g
    ⊢ Nat.Partrec' fun v => Part.map (fun a => g (List.Vector.cons a v)) (f v)
  -/
  simpa [(Part.bind_some_eq_map _ _).symm] using hf.bind hg
  /-
    🎉 no goals
  -/


/-- Analogous to `Nat.Partrec'` for `ℕ`-valued functions, a predicate for partial recursive
  vector-valued functions. -/
def Vec {n m} (f : List.Vector ℕ n → List.Vector ℕ m) :=
  ∀ i, Partrec' fun v => (f v).get i


nonrec theorem Vec.prim {n m f} (hf : @Nat.Primrec'.Vec n m f) : Vec f := fun i => prim <| hf i


protected theorem nil {n} : @Vec n 0 fun _ => nil := fun i => i.elim0


protected theorem cons {n m} {f : List.Vector ℕ n → ℕ} {g} (hf : @Partrec' n f)
    (hg : @Vec n m g) : Vec fun v => f v ::ᵥ g v := fun i =>
                /-
                  n m : Nat
                  f : List.Vector Nat n → Nat
                  g : List.Vector Nat n → List.Vector Nat m
                  hf : Nat.Partrec' ↑f
                  hg : Nat.Partrec'.Vec g
                  i : Fin m.succ
                  ⊢ Nat.Partrec' fun v => ↑(Option.some (((fun v => List.Vector.cons (f v) (g v) …
                -/
                /-
                  🎉 no goals
                -/
  Fin.cases (by simpa using hf) (fun i => by simp only [hg i, get_cons_succ]) i
                                             /-
                                               🎉 no goals
                                             -/


theorem idv {n} : @Vec n n id :=
  Vec.prim Nat.Primrec'.idv


theorem comp' {n m f g} (hf : @Partrec' m f) (hg : @Vec n m g) : Partrec' fun v => f (g v) :=
                                   /-
                                     n m : Nat
                                     f : PFun (List.Vector Nat m) Nat
                                     g : List.Vector Nat n → List.Vector Nat m
                                     hf : Nat.Partrec' f
                                     hg : Nat.Partrec'.Vec g
                                     v : List.Vector Nat n
                                     ⊢ Eq (Bind.bind (List.Vector.mOfFn fun i => ↑(Option.some ((g v).get i))) f) ( …
                                   -/
  (hf.comp _ hg).of_eq fun v => by simp
                                   /-
                                     🎉 no goals
                                   -/


theorem comp₁ {n} (f : ℕ →. ℕ) {g : List.Vector ℕ n → ℕ} (hf : @Partrec' 1 fun v => f v.head)
    (hg : @Partrec' n g) : @Partrec' n fun v => f (g v) := by
  /-
    n : Nat
    f : PFun Nat Nat
    g : List.Vector Nat n → Nat
    hf : Nat.Partrec' fun v => f v.head
    hg : Nat.Partrec' ↑g
    ⊢ Nat.Partrec' fun v => f (g v)
  -/
  simpa using hf.comp' (Partrec'.cons hg Partrec'.nil)
  /-
    🎉 no goals
  -/


theorem rfindOpt {n} {f : List.Vector ℕ (n + 1) → ℕ} (hf : @Partrec' (n + 1) f) :
    @Partrec' n fun v => Nat.rfindOpt fun a => ofNat (Option ℕ) (f (a ::ᵥ v)) :=
  ((rfind <|
        (of_prim (Primrec.nat_sub.comp (_root_.Primrec.const 1) Primrec.vector_head)).comp₁
          (fun n => Part.some (1 - n)) hf).bind
    ((prim Nat.Primrec'.pred).comp₁ Nat.pred hf)).of_eq
    fun v =>
    Part.ext fun b => by
      simp only [Nat.rfindOpt, exists_prop, tsub_eq_zero_iff_le, PFun.coe_val, Part.mem_bind_iff,
        Part.mem_some_iff, Option.mem_def, Part.mem_coe]
      refine
        exists_congr fun a => (and_congr (iff_of_eq ?_) Iff.rfl).trans (and_congr_right fun h => ?_)
        /-
          case refine_1
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          ⊢ Eq (Membership.mem (Nat.rfind fun n_1 => Part.some (Decidable.decide (LE.le  …
        -/
      · congr
        /-
          case refine_1.e_a.e_p
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          ⊢ Eq (fun n_1 => Part.some (Decidable.decide (LE.le 1 (f (List.Vector.cons n_1 …
        -/
        funext n
        /-
          case refine_1.e_a.e_p.h
          n✝ : Nat
          f : List.Vector Nat (HAdd.hAdd n✝ 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n✝
          b a n : Nat
          ⊢ Eq (Part.some (Decidable.decide (LE.le 1 (f (List.Vector.cons n v))))) ↑(Opt …
        -/
                                                          /-
                                                            🎉 no goals
                                                          -/
        cases f (n ::ᵥ v) <;> simp [Nat.succ_le_succ] <;> rfl
                                                          /-
                                                            🎉 no goals
                                                          -/
        /-
          case refine_2
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          h : Membership.mem (Nat.rfind fun n_1 => ↑(Option.some (Denumerable.ofNat (Opt …
          ⊢ Iff (Eq b (f (List.Vector.cons a v)).pred) (Eq (Denumerable.ofNat (Option Na …
        -/
      · have := Nat.rfind_spec h
        /-
          case refine_2
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          h : Membership.mem (Nat.rfind fun n_1 => ↑(Option.some (Denumerable.ofNat (Opt …
          this : Membership.mem (↑(Option.some (Denumerable.ofNat (Option Nat) (f (List. …
          ⊢ Iff (Eq b (f (List.Vector.cons a v)).pred) (Eq (Denumerable.ofNat (Option Na …
        -/
        simp only [Part.coe_some, Part.mem_some_iff] at this
        /-
          case refine_2
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          h : Membership.mem (Nat.rfind fun n_1 => ↑(Option.some (Denumerable.ofNat (Opt …
          this : Eq Bool.true (Denumerable.ofNat (Option Nat) (f (List.Vector.cons a v)) …
          ⊢ Iff (Eq b (f (List.Vector.cons a v)).pred) (Eq (Denumerable.ofNat (Option Na …
        -/
        revert this; cases' f (a ::ᵥ v) with c <;> intro this
          /-
            case refine_2.zero
            n : Nat
            f : List.Vector Nat (HAdd.hAdd n 1) → Nat
            hf : Nat.Partrec' ↑f
            v : List.Vector Nat n
            b a : Nat
            h : Membership.mem (Nat.rfind fun n_1 => ↑(Option.some (Denumerable.ofNat (Opt …
            this : Eq Bool.true (Denumerable.ofNat (Option Nat) 0).isSome
            ⊢ Iff (Eq b (Nat.pred 0)) (Eq (Denumerable.ofNat (Option Nat) 0) (Option.some  …
          -/
        · cases this
          /-
            🎉 no goals
          -/
        /-
          case refine_2.succ
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          h : Membership.mem (Nat.rfind fun n_1 => ↑(Option.some (Denumerable.ofNat (Opt …
          c : Nat
          this : Eq Bool.true (Denumerable.ofNat (Option Nat) (HAdd.hAdd c 1)).isSome
          ⊢ Iff (Eq b (HAdd.hAdd c 1).pred) (Eq (Denumerable.ofNat (Option Nat) (HAdd.hA …
        -/
        rw [← Option.some_inj, eq_comm]
        /-
          case refine_2.succ
          n : Nat
          f : List.Vector Nat (HAdd.hAdd n 1) → Nat
          hf : Nat.Partrec' ↑f
          v : List.Vector Nat n
          b a : Nat
          h : Membership.mem (Nat.rfind fun n_1 => ↑(Option.some (Denumerable.ofNat (Opt …
          c : Nat
          this : Eq Bool.true (Denumerable.ofNat (Option Nat) (HAdd.hAdd c 1)).isSome
          ⊢ Iff (Eq (Option.some (HAdd.hAdd c 1).pred) (Option.some b)) (Eq (Denumerable …
        -/
        rfl
        /-
          🎉 no goals
        -/


theorem of_part : ∀ {n f}, _root_.Partrec f → @Partrec' n f :=
  @(suffices ∀ f, Nat.Partrec f → @Partrec' 1 fun v => f v.head from fun {n f} hf => by
      let g := fun n₁ =>
        (Part.ofOption (decode (α := List.Vector ℕ n) n₁)).bind (fun a => Part.map encode (f a))
      exact
        (comp₁ g (this g hf) (prim Nat.Primrec'.encode)).of_eq fun i => by
          dsimp only [g]; simp [encodek, Part.map_id']
    fun f hf => by
    /-
      f : PFun Nat Nat
      hf : Nat.Partrec f
      ⊢ Nat.Partrec' fun v => f v.head
    -/
    obtain ⟨c, rfl⟩ := exists_code.1 hf
    simpa [eval_eq_rfindOpt] using
      rfindOpt <|
        of_prim <|
          Primrec.encode_iff.2 <|
            evaln_prim.comp <|
              (Primrec.vector_head.pair (_root_.Primrec.const c)).pair <|
                Primrec.vector_head.comp Primrec.vector_tail)


theorem part_iff {n f} : @Partrec' n f ↔ _root_.Partrec f :=
  ⟨to_part, of_part⟩


theorem part_iff₁ {f : ℕ →. ℕ} : (@Partrec' 1 fun v => f v.head) ↔ _root_.Partrec f :=
  part_iff.trans
    ⟨fun h =>
      (h.comp <| (Primrec.vector_ofFn fun _ => _root_.Primrec.id).to_comp).of_eq fun v => by
        /-
          f : PFun Nat Nat
          h : _root_.Partrec fun v => f v.head
          v : Nat
          ⊢ Eq (f (List.Vector.ofFn fun i => id v).head) (f v)
        -/
        simp only [id, head_ofFn],
        /-
          🎉 no goals
        -/
      fun h => h.comp vector_head⟩


theorem part_iff₂ {f : ℕ → ℕ →. ℕ} : (@Partrec' 2 fun v => f v.head v.tail.head) ↔ Partrec₂ f :=
  part_iff.trans
    ⟨fun h =>
      (h.comp <| vector_cons.comp fst <| vector_cons.comp snd (const nil)).of_eq fun v => by
        /-
          f : Nat → PFun Nat Nat
          h : _root_.Partrec fun v => f v.head v.tail.head
          v : Prod Nat Nat
          ⊢ Eq (f (List.Vector.cons v.1 (List.Vector.cons v.2 List.Vector.nil)).head (Li …
        -/
        simp only [head_cons, tail_cons],
        /-
          🎉 no goals
        -/
      fun h => h.comp vector_head (vector_head.comp vector_tail)⟩


theorem vec_iff {m n f} : @Vec m n f ↔ Computable f :=
               /-
                 m n : Nat
                 f : List.Vector Nat m → List.Vector Nat n
                 h : Nat.Partrec'.Vec f
                 ⊢ Computable f
               -/
  ⟨fun h => by simpa only [ofFn_get] using vector_ofFn fun i => to_part (h i), fun h i =>
               /-
                 🎉 no goals
               -/
    of_part <| vector_get.comp h (const i)⟩


